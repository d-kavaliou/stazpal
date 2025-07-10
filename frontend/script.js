class StazPalAudioService {
    constructor() {
        this.ws = null;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.sessionId = null;
        this.isRecording = false;
        this.isConnected = false;
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupVisualizer();
    }
    
    initializeElements() {
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.connectionText = document.getElementById('connectionText');
        this.recordingText = document.getElementById('recordingText');
        this.transcript = document.getElementById('transcript');
        this.visualizer = document.getElementById('visualizer');
        this.canvasCtx = this.visualizer.getContext('2d');
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        
        // Auto-connect on page load
        this.connectWebSocket();
    }
    
    setupVisualizer() {
        this.visualizer.width = 400;
        this.visualizer.height = 100;
        this.drawVisualizer();
    }
    
    connectWebSocket() {
        console.log('Connecting to WebSocket...');
        this.ws = new WebSocket('ws://localhost:3030');
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus('connected');
        };
        
        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleServerMessage(message);
            } catch (e) {
                console.error('Error parsing message:', e);
            }
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    this.connectWebSocket();
                }
            }, 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.isConnected = false;
            this.updateConnectionStatus('disconnected');
        };
    }
    
    handleServerMessage(message) {
        console.log('Received message:', message);
        
        switch (message.type) {
            case 'connected':
                this.sessionId = message.session_id;
                console.log('Session established:', this.sessionId);
                break;
                
            case 'text_chunk':
                this.displayTextChunk(message.text, message.is_final);
                break;
                
            case 'session_end':
                console.log('Session ended:', message.session_id);
                this.stopRecording();
                break;
                
            case 'error':
                console.error('Server error:', message.message);
                this.displayError(message.message);
                break;
        }
    }
    
    displayTextChunk(text, isFinal) {
        // Remove placeholder if it exists
        const placeholder = this.transcript.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Create text chunk element
        const textElement = document.createElement('span');
        textElement.className = isFinal ? 'text-chunk final' : 'text-chunk partial';
        textElement.textContent = text + ' ';
        
        this.transcript.appendChild(textElement);
        
        // Auto-scroll to bottom
        this.transcript.scrollTop = this.transcript.scrollHeight;
    }
    
    displayError(message) {
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = `Error: ${message}`;
        errorElement.style.color = 'red';
        errorElement.style.fontWeight = 'bold';
        errorElement.style.margin = '10px 0';
        
        this.transcript.appendChild(errorElement);
        this.transcript.scrollTop = this.transcript.scrollHeight;
    }
    
    async startRecording() {
        if (!this.isConnected) {
            alert('Please wait for WebSocket connection...');
            return;
        }
        
        if (this.isRecording) return;
        
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            // Setup audio context for visualization
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyser);
            
            this.analyser.fftSize = 256;
            this.startVisualization();
            
            // Setup MediaRecorder
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0 && this.isRecording) {
                    this.sendAudioChunk(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                stream.getTracks().forEach(track => track.stop());
                if (this.audioContext) {
                    this.audioContext.close();
                    this.audioContext = null;
                }
            };
            
            // Start recording
            this.mediaRecorder.start(250); // Send chunks every 250ms
            this.isRecording = true;
            
            // Send start message
            this.sendMessage({
                type: 'start',
                session_id: this.sessionId
            });
            
            this.updateRecordingStatus('recording');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            
            // Clear previous transcript
            this.transcript.innerHTML = '<p class="placeholder">Listening...</p>';
            
        } catch (error) {
            console.error('Error starting recording:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        
        this.isRecording = false;
        
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
        
        // Send end message
        this.sendMessage({
            type: 'end'
        });
        
        this.updateRecordingStatus('stopped');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        // Add final message to transcript
        const finalElement = document.createElement('div');
        finalElement.className = 'transcript-end';
        finalElement.textContent = '--- Recording stopped ---';
        finalElement.style.color = '#666';
        finalElement.style.fontStyle = 'italic';
        finalElement.style.textAlign = 'center';
        finalElement.style.margin = '20px 0';
        
        this.transcript.appendChild(finalElement);
        this.transcript.scrollTop = this.transcript.scrollHeight;
    }
    
    async sendAudioChunk(audioBlob) {
        if (!this.isConnected || !this.isRecording) return;
        
        try {
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64Data = this.arrayBufferToBase64(arrayBuffer);
            
            this.sendMessage({
                type: 'audio_chunk',
                data: base64Data
            });
        } catch (error) {
            console.error('Error sending audio chunk:', error);
        }
    }
    
    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }
    
    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }
    
    updateConnectionStatus(status) {
        this.connectionText.textContent = status === 'connected' ? 'Connected' : 'Disconnected';
        this.connectionText.className = `status-value ${status}`;
    }
    
    updateRecordingStatus(status) {
        this.recordingText.textContent = status === 'recording' ? 'Recording...' : 'Stopped';
        this.recordingText.className = `status-value ${status}`;
    }
    
    startVisualization() {
        if (!this.analyser) return;
        
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            if (!this.isRecording) {
                this.drawVisualizer();
                return;
            }
            
            requestAnimationFrame(draw);
            
            this.analyser.getByteFrequencyData(dataArray);
            
            this.canvasCtx.fillStyle = '#f8f9fa';
            this.canvasCtx.fillRect(0, 0, this.visualizer.width, this.visualizer.height);
            
            const barWidth = (this.visualizer.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 255 * this.visualizer.height;
                
                this.canvasCtx.fillStyle = `hsl(${barHeight * 2}, 70%, 50%)`;
                this.canvasCtx.fillRect(x, this.visualizer.height - barHeight, barWidth, barHeight);
                
                x += barWidth + 1;
            }
        };
        
        draw();
    }
    
    drawVisualizer() {
        this.canvasCtx.fillStyle = '#f8f9fa';
        this.canvasCtx.fillRect(0, 0, this.visualizer.width, this.visualizer.height);
        
        // Draw idle state
        this.canvasCtx.fillStyle = '#e9ecef';
        this.canvasCtx.fillRect(0, this.visualizer.height / 2 - 1, this.visualizer.width, 2);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new StazPalAudioService();
});