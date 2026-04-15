# ============================================
# DEEPFAKE AUDIO DETECTION & ENHANCEMENT SYSTEM
# WORKS WITH ANY AUDIO FORMAT (M4A, MP3, WAV, etc.)
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
import subprocess
import sys

warnings.filterwarnings('ignore')

# ============================================
# YOUR PUBLIC GOOGLE DRIVE FILE ID
# ============================================
FILE_ID = "14zV9ptEwXzI_0yOAyjrZe_4QjeggIPjJ"

# ============================================
# COMPLETE RESNET MODEL
# ============================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = torch.relu(out)
        return out

class ResNetDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ============================================
# AUDIO PREPROCESSOR - HANDLES ANY FORMAT
# ============================================

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, duration=3.0, n_mels=128):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = 1024
        self.hop_length = 512
        self.max_length = int(sample_rate * duration)
    
    def convert_to_wav(self, input_path, output_path):
        """Convert any audio format to WAV using ffmpeg"""
        try:
            # Try using ffmpeg (most reliable)
            cmd = f"ffmpeg -i {input_path} -ar {self.sample_rate} -ac 1 {output_path} -y -loglevel quiet"
            subprocess.run(cmd, shell=True, check=True)
            return True
        except:
            try:
                # Fallback: Use librosa directly (supports m4a, mp3, etc.)
                audio, sr = librosa.load(input_path, sr=self.sample_rate, mono=True, duration=self.duration)
                sf.write(output_path, audio, self.sample_rate)
                return True
            except:
                return False
    
    def load_audio(self, audio_path):
        """Load audio from any format"""
        try:
            # Try direct loading with librosa (supports m4a, mp3, wav)
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True, duration=self.duration)
            return audio, sr
        except:
            try:
                # Convert to wav first
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                if self.convert_to_wav(audio_path, temp_wav):
                    audio, sr = librosa.load(temp_wav, sr=self.sample_rate, mono=True)
                    os.unlink(temp_wav)
                    return audio, sr
            except:
                pass
            raise Exception(f"Cannot load audio file: {audio_path}")
    
    def pad_or_truncate(self, audio):
        if len(audio) >= self.max_length:
            return audio[:self.max_length]
        else:
            return np.pad(audio, (0, self.max_length - len(audio)))
    
    def extract_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel_spec)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel
    
    def process_audio(self, audio_path):
        audio, _ = self.load_audio(audio_path)
        audio = self.pad_or_truncate(audio)
        mel_spec = self.extract_mel_spectrogram(audio)
        return mel_spec, audio

# ============================================
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# ============================================

@st.cache_resource
def load_model():
    model_path = Path("deepfake_detector.pth")
    
    if not model_path.exists():
        with st.spinner("📥 Downloading AI model (128MB)... First time only, please wait..."):
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, str(model_path), quiet=False)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Download failed: {e}")
                return None
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = ResNetDetector(num_classes=2).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None

# ============================================
# DEEPFAKE DETECTOR
# ============================================

class DeepfakeDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.preprocessor = AudioPreprocessor()
    
    def predict(self, audio_path):
        mel_spec, audio = self.preprocessor.process_audio(audio_path)
        input_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).item()
            confidence = probs[0, pred].item()
        
        return {
            'prediction': 'REAL' if pred == 0 else 'DEEPFAKE',
            'confidence': confidence,
            'real_prob': probs[0, 0].item(),
            'fake_prob': probs[0, 1].item(),
            'mel_spec': mel_spec,
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
    .title { font-size: 2.5rem; font-weight: bold; text-align: center; 
             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .real-badge { background-color: #10b981; color: white; padding: 0.75rem; border-radius: 2rem; text-align: center; font-weight: bold; }
    .fake-badge { background-color: #ef4444; color: white; padding: 0.75rem; border-radius: 2rem; text-align: center; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎵 Deepfake Audio Detection System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload Audio")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'wma']
    )
    st.markdown("---")
    st.header("⚙️ Settings")
    auto_enhance = st.checkbox("Auto-enhance deepfake audio", value=True)
    show_spectrogram = st.checkbox("Show spectrogram", value=True)

# Main content
if uploaded_file is not None:
    # Load model
    model, device = load_model()
    
    if model is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Display original audio
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎧 Original Audio")
            st.audio(temp_path, format='audio/wav')
        
        # Detect
        detector = DeepfakeDetector(model, device)
        
        with st.spinner("🔍 Analyzing audio..."):
            try:
                result = detector.predict(temp_path)
                
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
                st.progress(result['real_prob'], text=f"🎵 Real: {result['real_prob']*100:.1f}%")
                st.progress(result['fake_prob'], text=f"🤖 Deepfake: {result['fake_prob']*100:.1f}%")
                
                # Spectrogram
                if show_spectrogram:
                    st.subheader("📊 Spectrogram")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(result['mel_spec'], aspect='auto', origin='lower', cmap='viridis')
                    ax.set_title(f"Mel-Spectrogram - {result['prediction']}")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                
                # Enhancement for deepfake
                if result['prediction'] == 'DEEPFAKE' and auto_enhance:
                    st.markdown("---")
                    st.subheader("🎛️ Enhanced Audio")
                    enhanced_path = temp_path + "_enhanced.wav"
                    AudioEnhancer.enhance_file(temp_path, enhanced_path)
                    with col2:
                        st.audio(enhanced_path)
                        with open(enhanced_path, 'rb') as f:
                            st.download_button("💾 Download Enhanced", f, file_name="enhanced_audio.wav")
                
                # Download original
                with open(temp_path, 'rb') as f:
                    st.download_button("💾 Download Original", f, file_name=uploaded_file.name)
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                st.info("Please try a different audio file or convert to WAV format")
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

else:
    st.info("👈 **Please upload an audio file (MP3, M4A, WAV, etc.)**")
    
    st.markdown("---")
    st.subheader("✨ Supported Formats")
    st.write("✅ MP3, M4A, WAV, FLAC, OGG, AAC, WMA")
    st.write("✅ Any sample rate (auto-converted to 16kHz)")
    st.write("✅ Any duration (trimmed/padded to 3 seconds)")
