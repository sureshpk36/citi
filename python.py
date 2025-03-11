from flask import Flask, render_template, request, jsonify
import requests
import os
import tempfile
import base64
import time
import threading
import queue
import re
import speech_recognition as sr
from gtts import gTTS
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Mistral API Configuration
MISTRAL_API_KEY = "XZDVrVnzsZTnakgniECWmP9OS6QjhaiY"
MISTRAL_MODEL = "mistral-medium"
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Queue for speech processing; items will be tuples of (token, sentence)
tts_queue = queue.Queue()

# Global token to keep track of the current response
current_token = 0

# Function to Fetch Response from Mistral API
def get_medical_response(user_input):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI medical assistant. Provide your response in clear, short sentences separated by periods. First tell the user what you're going to explain, then provide the information. If symptoms are mentioned, suggest possible conditions and first-aid remedies. Recommend only OTC (over-the-counter) medicines. If symptoms are severe, suggest consulting a doctor."},
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    try:
        response = requests.post(MISTRAL_ENDPOINT, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error: Unable to fetch response. {str(e)}"

def stream_response(user_input, token):
    global current_token
    if token != current_token:
        return  # Exit if this response is no longer current
    
    socketio.emit('thinking_status', {'status': True})
    
    try:
        full_response = get_medical_response(user_input)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', full_response) if s.strip()]
        
        print(f"Processing response with {len(sentences)} sentences")
        
        accumulated_text = ""
        for sentence in sentences:
            if token != current_token:
                print("Token changed, stopping response streaming")
                break  # Stop processing if token has changed
            
            accumulated_text += (" " if accumulated_text else "") + sentence
            socketio.emit('response_stream', {'text': accumulated_text, 'is_final': False})
            
            print(f"Adding sentence to TTS queue: '{sentence}'")
            tts_queue.put((token, sentence))
            
            socketio.sleep(0.3)  # Reduced delay for quicker response
            
        if token == current_token:
            socketio.emit('response_stream', {'text': accumulated_text, 'is_final': True})
            
    except Exception as e:
        print(f"Error in stream_response: {str(e)}")
        socketio.emit('error_message', {'message': f'Error generating response: {str(e)}'})
    finally:
        if token == current_token:
            socketio.emit('thinking_status', {'status': False})

# Function to handle speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    
    try:
        socketio.emit('listening_status', {'status': True})
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        socketio.emit('listening_status', {'status': False})
        
        # Convert Speech to Text
        user_input = recognizer.recognize_google(audio)
        
        # Emit recognized speech
        socketio.emit('speech_recognized', {'text': user_input})
        
        # Process the response in a new thread with the current token
        global current_token
        thread = threading.Thread(target=stream_response, args=(user_input, current_token))
        thread.daemon = True
        thread.start()
        
    except sr.UnknownValueError:
        socketio.emit('listening_status', {'status': False})
        socketio.emit('error_message', {'message': 'Sorry, I couldn\'t understand. Please try again.'})
    except sr.RequestError as e:
        socketio.emit('listening_status', {'status': False})
        socketio.emit('error_message', {'message': f'Error in speech recognition service: {str(e)}'})
    except Exception as e:
        socketio.emit('listening_status', {'status': False})
        socketio.emit('error_message', {'message': f'An error occurred: {str(e)}'})

# Function to generate speech audio optimized for sentence-by-sentence processing
def text_to_speech(text):
    try:
        if not text or len(text.strip()) < 2:
            print("Empty text received, skipping TTS")
            return
            
        print(f"Converting to speech: '{text}'")
        
        # Create temp file in a way that ensures it's accessible
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_filename = temp_file.name
            print(f"Temp file created: {temp_filename}")
        
        # Generate and save audio
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_filename)
        print("Audio saved to temp file")
        
        # Add a small delay to ensure file is fully written
        time.sleep(0.2)
        
        # Read the audio file
        with open(temp_filename, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            print(f"Audio file read, size: {len(audio_data)} chars")
        
        # Clean up
        try:
            os.unlink(temp_filename)
            print("Temp file deleted")
        except Exception as e:
            print(f"Warning: Could not delete temp file: {str(e)}")
        
        # Send the audio data to the client
        print("Sending audio data to client")
        socketio.emit('play_audio', {'audio_data': audio_data})
        print("Audio data sent")
    
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        socketio.emit('error_message', {'message': f'Error generating speech: {str(e)}'})

# TTS worker thread: processes items from the queue if they belong to the current message
def tts_worker():
    global current_token
    print("TTS worker thread started")
    while True:
        try:
            print("Waiting for sentence in TTS queue...")
            token, text = tts_queue.get()
            print(f"Got sentence from queue (token {token}): '{text}'")
            
            if token != current_token:
                print(f"Skipping TTS for outdated token {token} (current token is {current_token})")
                tts_queue.task_done()
                continue
                
            text_to_speech(text)
            tts_queue.task_done()
        except Exception as e:
            print(f"Error in TTS worker: {str(e)}")
            try:
                tts_queue.task_done()
            except:
                pass

# Start TTS worker thread
print("Starting TTS worker thread")
tts_thread = threading.Thread(target=tts_worker)
tts_thread.daemon = True
tts_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test_tts')
def test_tts():
    print("Testing TTS functionality")
    text_to_speech("This is a test of the text to speech system.")
    return "Testing TTS functionality. Check console for logs."

@socketio.on('send_message')
def handle_message(data):
    global current_token, tts_queue
    user_input = data['message'].strip()
    if not user_input:
        return
    
    print(f"Received new message: '{user_input}'")
    
    # Cancel previous processing
    current_token += 1
    print(f"Cancelling previous processing, new token: {current_token}")
    
    with tts_queue.mutex:
        queue_size = len(tts_queue.queue)
        tts_queue.queue.clear()
        print(f"Cleared TTS queue ({queue_size} items removed)")
    
    # Notify client to stop audio immediately
    socketio.emit('stop_audio')
    print("Sent stop_audio signal to client")
    
    # Start new processing thread
    print(f"Starting new processing thread for token {current_token}")
    thread = threading.Thread(target=stream_response, args=(user_input, current_token))
    thread.daemon = True
    thread.start()

@socketio.on('start_voice_input')
def handle_voice_input():
    print("Starting voice input")
    thread = threading.Thread(target=recognize_speech)
    thread.daemon = True
    thread.start()

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    
    # Updated HTML template with improved audio handling and debugging
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body { background-color: #f8f9fa; font-family: 'Arial', sans-serif; }
        .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; background-color: white; border-radius: 15px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1); display: flex; flex-direction: column; height: 85vh; }
        .chat-header { background: linear-gradient(135deg, #43a047, #2e7d32); color: white; padding: 15px; border-radius: 10px 10px 0 0; font-size: 1.5rem; text-align: center; margin-bottom: 15px; }
        .chat-messages { flex-grow: 1; overflow-y: auto; padding: 10px; margin-bottom: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
        .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 10px; max-width: 75%; word-wrap: break-word; }
        .user-message { background-color: #e3f2fd; color: #0d47a1; margin-left: auto; border-bottom-right-radius: 0; }
        .bot-message { background-color: #e8f5e9; color: #1b5e20; margin-right: auto; border-bottom-left-radius: 0; }
        .system-message { background-color: #f5f5f5; color: #757575; text-align: center; margin: 10px auto; font-style: italic; max-width: 50%; }
        .thinking { display: flex; align-items: center; margin-bottom: 15px; }
        .thinking-dots { display: flex; margin-left: 10px; }
        .thinking-dot { height: 10px; width: 10px; margin: 0 3px; background-color: #1b5e20; border-radius: 50%; animation: pulse 1.5s infinite; }
        .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
        .input-container { display: flex; margin-top: 10px; }
        .message-input { flex-grow: 1; padding: 12px; border: 1px solid #ccc; border-radius: 25px; font-size: 1rem; outline: none; transition: border 0.3s; }
        .message-input:focus { border-color: #43a047; }
        .send-button, .voice-button { padding: 10px 15px; margin-left: 10px; border: none; border-radius: 25px; cursor: pointer; outline: none; transition: background-color 0.3s; }
        .send-button { background-color: #43a047; color: white; }
        .voice-button { background-color: #2196f3; color: white; }
        .send-button:hover { background-color: #2e7d32; }
        .voice-button:hover { background-color: #1976d2; }
        .voice-button.listening { animation: pulse 1.5s infinite; background-color: #f44336; }
        @keyframes pulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.1); opacity: 0.7; } 100% { transform: scale(1); opacity: 1; } }
        .status-bar { margin-top: 15px; padding: 5px 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 0.85rem; color: #757575; }
        @media (max-width: 576px) { .chat-container { height: 95vh; border-radius: 0; box-shadow: none; } .message { max-width: 85%; } }
        .debug-panel { padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-top: 10px; font-family: monospace; font-size: 0.8rem; max-height: 100px; overflow-y: auto; display: none; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="chat-container">
            <div class="chat-header">
                <i class="fas fa-notes-medical me-2"></i> Medical Assistant
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="message bot-message">
                    Hello! I'm your medical assistant. How can I help you today?
                </div>
            </div>
            <div class="input-container">
                <input type="text" class="message-input" id="messageInput" placeholder="Type your message here...">
                <button class="send-button" id="sendButton">
                    <i class="fas fa-paper-plane"></i>
                </button>
                <button class="voice-button" id="voiceButton">
                    <i class="fas fa-microphone"></i>
                </button>
            </div>
            <div class="status-bar" id="statusBar">Ready</div>
            <div class="debug-panel" id="debugPanel"></div>
        </div>
    </div>
    <script src="https://cdn.socket.io/4.4.1/socket.io.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const socket = io();
            const chatMessages = document.getElementById('chatMessages');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const voiceButton = document.getElementById('voiceButton');
            const statusBar = document.getElementById('statusBar');
            const debugPanel = document.getElementById('debugPanel');
            
            // Uncomment to enable debug panel
            // debugPanel.style.display = 'block';
            
            let isThinking = false;
            let isListening = false;
            let currentBotMessage = null;
            let currentThinking = null;
            let audioQueue = [];
            let isPlayingAudio = false;
            let currentAudio = null;
            let audioContext = null;
            
            // Debug log function
            function debugLog(message) {
                console.log(message);
                const timestamp = new Date().toISOString().substr(11, 8);
                debugPanel.innerHTML += `<div>[${timestamp}] ${message}</div>`;
                debugPanel.scrollTop = debugPanel.scrollHeight;
            }
            
            // Initialize audio context with user interaction
            function initAudioContext() {
                if (!audioContext) {
                    try {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                        debugLog("Audio context initialized");
                    } catch (e) {
                        debugLog("Error initializing audio context: " + e);
                    }
                }
            }
            
            function sendMessage() {
                const message = messageInput.value.trim();
                if (message && !isThinking && !isListening) {
                    initAudioContext(); // Initialize audio context with user interaction
                    
                    const userMessageElement = document.createElement('div');
                    userMessageElement.className = 'message user-message';
                    userMessageElement.textContent = message;
                    chatMessages.appendChild(userMessageElement);
                    messageInput.value = '';
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    
                    debugLog("Sending message: " + message);
                    socket.emit('send_message', { message: message });
                }
            }
            
            function showThinking() {
                const thinkingElement = document.createElement('div');
                thinkingElement.className = 'thinking';
                thinkingElement.innerHTML = `
                    <div class="message bot-message" style="margin-bottom: 0; padding-right: 20px;">
                        <div class="thinking-dots">
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                        </div>
                    </div>
                `;
                chatMessages.appendChild(thinkingElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                return thinkingElement;
            }
            
            sendButton.addEventListener('click', sendMessage);
            
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            voiceButton.addEventListener('click', function() {
                if (!isListening && !isThinking) {
                    initAudioContext(); // Initialize audio context with user interaction
                    debugLog("Starting voice input");
                    socket.emit('start_voice_input');
                }
            });
            
            // Test button for debugging
            document.addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.shiftKey && e.key === 'T') {
                    fetch('/test_tts')
                        .then(response => response.text())
                        .then(data => debugLog("TTS Test: " + data))
                        .catch(error => debugLog("TTS Test Error: " + error));
                }
            });
            
            socket.on('connect', function() {
                debugLog("Connected to server");
            });
            
            socket.on('disconnect', function() {
                debugLog("Disconnected from server");
            });
            
            socket.on('thinking_status', function(data) {
                isThinking = data.status;
                debugLog("Thinking status: " + data.status);
                if (isThinking) {
                    statusBar.textContent = 'Thinking...';
                    currentThinking = showThinking();
                } else {
                    statusBar.textContent = 'Ready';
                }
            });
            
            socket.on('listening_status', function(data) {
                isListening = data.status;
                debugLog("Listening status: " + data.status);
                if (isListening) {
                    voiceButton.classList.add('listening');
                    const listeningElement = document.createElement('div');
                    listeningElement.className = 'message system-message';
                    listeningElement.id = 'listeningMessage';
                    listeningElement.textContent = 'Listening...';
                    chatMessages.appendChild(listeningElement);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    statusBar.textContent = 'Listening...';
                } else {
                    voiceButton.classList.remove('listening');
                    const listeningElement = document.getElementById('listeningMessage');
                    if (listeningElement) { listeningElement.remove(); }
                    statusBar.textContent = 'Ready';
                }
            });
            
            socket.on('speech_recognized', function(data) {
                debugLog("Speech recognized: " + data.text);
                const userMessageElement = document.createElement('div');
                userMessageElement.className = 'message user-message';
                userMessageElement.textContent = data.text;
                chatMessages.appendChild(userMessageElement);
                messageInput.value = data.text;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });
            
            socket.on('response_stream', function(data) {
                debugLog("Response stream: " + (data.is_final ? "final" : "partial"));
                if (currentThinking) {
                    currentThinking.remove();
                    currentThinking = null;
                }
                if (!currentBotMessage) {
                    currentBotMessage = document.createElement('div');
                    currentBotMessage.className = 'message bot-message';
                    chatMessages.appendChild(currentBotMessage);
                }
                currentBotMessage.textContent = data.text;
                chatMessages.scrollTop = chatMessages.scrollHeight;
                if (data.is_final) { 
                    debugLog("Response complete");
                    currentBotMessage = null; 
                }
            });
            
            socket.on('error_message', function(data) {
                debugLog("Error: " + data.message);
                const errorElement = document.createElement('div');
                errorElement.className = 'message system-message';
                errorElement.textContent = data.message;
                chatMessages.appendChild(errorElement);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                statusBar.textContent = 'Ready';
            });
            
            // Audio playback handling with improved logging
            socket.on('play_audio', function(data) {
                debugLog("Received audio data: " + data.audio_data.substring(0, 20) + "... (" + data.audio_data.length + " chars)");
                
                const audioSrc = 'data:audio/mp3;base64,' + data.audio_data;
                audioQueue.push(audioSrc);
                
                debugLog("Added to audio queue. Queue length: " + audioQueue.length);
                
                if (!isPlayingAudio) {
                    playNextInQueue();
                }
            });
            
            // Stop audio event handler: clears the queue and stops current playback
            socket.on('stop_audio', function() {
                debugLog("Received stop_audio signal");
                const queueLength = audioQueue.length;
                audioQueue = [];
                
                if (currentAudio) {
                    debugLog("Stopping current audio playback");
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                    currentAudio = null;
                }
                
                debugLog(`Audio stopped, cleared queue (${queueLength} items)`);
                isPlayingAudio = false;
            });
            
            function playNextInQueue() {
                if (audioQueue.length === 0) {
                    debugLog("Audio queue empty, playback complete");
                    isPlayingAudio = false;
                    return;
                }
                
                isPlayingAudio = true;
                const nextAudioSrc = audioQueue.shift();
                
                debugLog("Playing next audio in queue, remaining: " + audioQueue.length);
                
                try {
                    currentAudio = new Audio(nextAudioSrc);
                    
                    currentAudio.oncanplaythrough = function() {
                        debugLog("Audio ready to play");
                        try {
                            const playPromise = currentAudio.play();
                            if (playPromise !== undefined) {
                                playPromise.catch(e => {
                                    debugLog("Playback failed with promise error: " + e);
                                    currentAudio = null;
                                    setTimeout(playNextInQueue, 200);
                                });
                            }
                        } catch (e) {
                            debugLog("Play error: " + e);
                            currentAudio = null;
                            setTimeout(playNextInQueue, 200);
                        }
                    };
                    
                    currentAudio.onended = function() {
                        debugLog("Audio playback completed");
                        currentAudio = null;
                        setTimeout(playNextInQueue, 200);
                    };
                    
                    currentAudio.onerror = function(e) {
                        debugLog("Audio playback error: " + e);
                        currentAudio = null;
                        setTimeout(playNextInQueue, 200);
                    };
                    
                    currentAudio.volume = 1.0;
                    
                } catch (e) {
                    debugLog("Audio creation error: " + e);
                    currentAudio = null;
                    setTimeout(playNextInQueue, 200);
                }
            }
            
            // Initialize UI
            debugLog("Medical Assistant UI initialized");
        });
    </script>
</body>
</html>
    '''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("Starting Medical Assistant Web App...")
    print("Open your browser and go to http://127.0.0.1:5000")
    socketio.run(app, debug=True)
