/**
 * Main JavaScript file for Speaker Recognition System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize counters
    initCounters();
    
    // Add active class to current nav item
    setActiveNavItem();
    
    // Initialize any audio visualizations
    initAudioVisualizations();

    // Check if audio upload form exists on the page
    const uploadForm = document.getElementById('audio-upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleAudioUpload);
    }

    // Initialize any tooltips
    initTooltips();
});

/**
 * Initialize counter animation for statistics
 */
function initCounters() {
    const counters = document.querySelectorAll('.counter');
    
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-target'));
        const duration = 1500; // ms
        const stepTime = 20; // ms
        
        let current = 0;
        const step = Math.max(1, Math.floor(target / (duration / stepTime)));
        
        const timer = setInterval(() => {
            current += step;
            
            if (current > target) {
                counter.textContent = target;
                clearInterval(timer);
            } else {
                counter.textContent = current;
            }
        }, stepTime);
    });
}

/**
 * Set active class on the current navigation item
 */
function setActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar .nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        
        if (currentPath === linkPath || 
            (currentPath === '/' && linkPath === '/') || 
            (currentPath !== '/' && linkPath !== '/' && currentPath.includes(linkPath))) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * Initialize audio visualizations for the audio player
 */
function initAudioVisualizations() {
    const audioPlayer = document.getElementById('audio-player');
    if (!audioPlayer) return;
    
    // Add event listener for the audio player
    audioPlayer.addEventListener('play', createVisualization);
}

/**
 * Create audio visualization if Web Audio API is available
 */
function createVisualization() {
    const audioPlayer = document.getElementById('audio-player');
    if (!audioPlayer) return;
    
    // Check for Web Audio API support
    if (!window.AudioContext && !window.webkitAudioContext) return;
    
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    const source = audioCtx.createMediaElementSource(audioPlayer);
    
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
    
    // Configure analyser
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    // Remove event listener to avoid creating multiple analyzers
    audioPlayer.removeEventListener('play', createVisualization);
}

/**
 * Format time in seconds to MM:SS format
 * @param {number} seconds 
 * @returns {string} Formatted time
 */
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}

/**
 * Handle file upload validation for audio files
 * @param {HTMLInputElement} fileInput 
 * @returns {boolean} Is valid file
 */
function validateAudioFile(fileInput) {
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('Please select an audio file');
        return false;
    }
    
    const file = fileInput.files[0];
    const fileType = file.type;
    
    // Check if file is a WAV audio file
    if (!fileType.startsWith('audio/')) {
        showError('Please upload an audio file');
        return false;
    }
    
    if (file.size > 16 * 1024 * 1024) { // 16MB
        showError('File size exceeds maximum limit (16MB)');
        return false;
    }
    
    return true;
}

/**
 * Show error message
 * @param {string} message 
 */
function showError(message) {
    const resultArea = document.getElementById('result-area');
    if (resultArea) {
        resultArea.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-circle me-2"></i> ${message}
            </div>
        `;
    }
}

/**
 * Handle audio file upload and prediction
 * @param {Event} event 
 */
function handleAudioUpload(event) {
    event.preventDefault();
    
    // Show loading spinner
    const resultArea = document.getElementById('result-area');
    if (resultArea) {
        resultArea.innerHTML = '<div class="text-center my-4"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Processing audio...</p></div>';
    }
    
    // Get form data
    const formData = new FormData(event.target);
    
    // Check if file is selected
    const audioFile = formData.get('audio_file');
    if (!audioFile || audioFile.size === 0) {
        showError('Please select an audio file');
        return;
    }
    
    // Make AJAX request
    fetch('/upload', {
        method: 'POST',
        body: formData,
        // Do not set Content-Type header when sending FormData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displayPredictionResult(data);
        } else {
            showError(data.error || 'Unknown error occurred');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Network error: ' + error.message);
    });
}

/**
 * Display prediction result in the UI
 * @param {Object} data 
 */
function displayPredictionResult(data) {
    const resultArea = document.getElementById('result-area');
    if (resultArea) {
        const confidenceRounded = Math.round(data.confidence * 10) / 10;
        
        resultArea.innerHTML = `
            <div class="card shadow-sm">
                <div class="card-body">
                    <h5 class="card-title text-success"><i class="fas fa-check-circle me-2"></i>Speaker Identified!</h5>
                    <p class="card-text">
                        <strong>Speaker identified as:</strong> ${data.speaker}<br>
                        <strong>Confidence score:</strong> ${confidenceRounded}%
                    </p>
                </div>
            </div>
        `;
        
        // Log the actual data received for debugging
        console.log('Prediction data received:', data);
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}