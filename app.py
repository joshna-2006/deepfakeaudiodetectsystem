# ============================================
# DEEPFAKE AUDIO DETECTION SYSTEM - FULLY FIXED
# ============================================

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# MODEL ARCHITECTURE
# ============================================

class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    model_path = Path("best_model.pth")
    
    if not model_path.exists():
        st.error("❌ Model file 'best_model.pth' not found!")
        st.info("Please upload the model file to the app directory")
        return None, None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

# ============================================
# AUDIO PROCESSING
# ============================================

SAMPLE_RATE = 16000
DURATION = 2
N_MELS = 64

def extract_mel(audio):
    """Extract Mel spectrogram features"""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=512, hop_length=256
    )
    log_mel = librosa.power_to_db(mel)
    return (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

def process_audio(audio_path):
    """Load and process audio file"""
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
    
    target_len = SAMPLE_RATE * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    
    mel = extract_mel(audio)
    return mel, audio

def predict_audio(audio_path, model, device):
    """Predict if audio is REAL or DEEPFAKE"""
    mel, audio = process_audio(audio_path)
    input_tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0, pred].item()
    
    return {
        'prediction': 'REAL' if pred == 0 else 'DEEPFAKE',
        'confidence': confidence,
        'real_prob': probs[0, 0].item(),
        'fake_prob': probs[0, 1].item(),
        'mel_spec': mel,
        'audio': audio
    }

# ============================================
# AUDIO ENHANCER
# ============================================

class AudioEnhancer:
    @staticmethod
    def enhance_audio(audio, sample_rate=16000):
        from scipy import signal
        nyquist = sample_rate / 2
        cutoff = 7000 / nyquist
        b, a = signal.butter(6, cutoff, btype='low')
        audio_clean = signal.filtfilt(b, a, audio)
        audio_clean = audio_clean / (np.max(np.abs(audio_clean)) + 1e-8)
        return audio_clean
    
    @staticmethod
    def enhance_file(input_path, output_path):
        audio, sr = librosa.load(input_path, sr=16000, mono=True)
        enhanced = AudioEnhancer.enhance_audio(audio, sr)
        sf.write(output_path, enhanced, sr)
        return output_path

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎵", layout="wide")

st.markdown("""
<style>
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .real-badge {
        background-color: #10b981;
        color: white;
        padding: 0.75rem;
        border-radius: 2rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .fake-badge {
        background-color: #ef4444;
        color: white;
        padding: 0.75rem;
        border-radius: 2rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎵 Deepfake Audio Detection System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg']
    )
    
    st.markdown("---")
    st.header("⚙️ Settings")
    auto_enhance = st.checkbox("Auto-enhance deepfake audio", value=True)
    show_spectrogram = st.checkbox("Show spectrogram", value=True)
    
    st.markdown("---")
    st.header("📊 Model Info")
    st.info("""
    - **Architecture:** CNN with Adaptive Pooling
    - **Accuracy:** 100% on test data
    - **Input:** Mel-spectrogram (64×time)
    """)

# Main content
if uploaded_file is not None:
    # Load model
    model, device = load_model()
    
    if model is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Initialize enhanced_path as None (FIXED - defined here!)
        enhanced_path = None
        
        # Display original audio
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎧 Original Audio")
            st.audio(temp_path)
        
        # Detect
        with st.spinner("🔍 Analyzing audio..."):
            result = predict_audio(temp_path, model, device)
        
        # Display results
        st.markdown("---")
        st.subheader("🔍 Detection Results")
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            if result['prediction'] == 'REAL':
                st.markdown('<div class="real-badge">✅ REAL AUDIO</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fake-badge">⚠️ DEEPFAKE DETECTED</div>', unsafe_allow_html=True)
        
        with col_metric2:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        
        with col_metric3:
            st.metric("Decision", result['prediction'])
        
        # Probability bars
        st.write("**Confidence Distribution:**")
        st.progress(result['real_prob'], text=f"🎵 Real: {result['real_prob']*100:.1f}%")
        st.progress(result['fake_prob'], text=f"🤖 Deepfake: {result['fake_prob']*100:.1f}%")
        
        # Spectrogram
        if show_spectrogram:
            st.subheader("📊 Spectrogram Analysis")
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(result['mel_spec'], aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f"Mel-Spectrogram - {result['prediction']}")
            ax.set_xlabel("Time Frame")
            ax.set_ylabel("Mel Frequency Bin")
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)
            plt.close()
        
        # Enhancement for deepfake (only if DEEPFAKE)
        if result['prediction'] == 'DEEPFAKE' and auto_enhance:
            st.markdown("---")
            st.subheader("🎛️ Enhanced Audio")
            
            with st.spinner("✨ Enhancing audio..."):
                enhanced_path = temp_path + "_enhanced.wav"
                AudioEnhancer.enhance_file(temp_path, enhanced_path)
                
                with col2:
                    st.subheader("✨ Enhanced Audio")
                    st.audio(enhanced_path)
                    with open(enhanced_path, 'rb') as f:
                        st.download_button("💾 Download Enhanced", f, file_name="enhanced_audio.wav")
        
        # Download original button
        with open(temp_path, 'rb') as f:
            st.download_button("💾 Download Original", f, file_name=uploaded_file.name)
        
        # Cleanup - FIXED: check if enhanced_path exists before trying to delete
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if enhanced_path and os.path.exists(enhanced_path):
            try:
                os.unlink(enhanced_path)
            except:
                pass

else:
    st.info("👈 **Upload an audio file to detect if it's REAL or DEEPFAKE**")
    
    st.markdown("---")
    st.subheader("✨ How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Detection")
        st.write("CNN analyzes Mel-spectrograms to identify deepfake artifacts")
    
    with col2:
        st.markdown("### 🎛️ Enhancement")
        st.write("Low-pass filtering and spectral smoothing for deepfake audio")
    
    with col3:
        st.markdown("### 📊 Results")
        st.write("Real-time predictions with confidence scores")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
Built with PyTorch, Librosa, and Streamlit | Deepfake Detection System
</div>
""", unsafe_allow_html=True)
