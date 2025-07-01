"""
Configuration management for AskMe Voice Assistant
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """LLM model configuration"""
    name: str = "mistral-7b-instruct"
    path: str = "./models/mistral-7b-instruct-gguf"
    model_file: str = "askme-q4_0.gguf"
    context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    device: str = "auto"
    gpu_layers: int = 35
    threads: int = -1
    batch_size: int = 512


@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration"""
    model: str = "base"
    language: Optional[str] = "en"
    task: str = "transcribe"
    device: str = "auto"
    compute_type: str = "float16"
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    condition_on_previous_text: bool = True


@dataclass
class TTSConfig:
    """Text-to-Speech configuration"""
    model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    vocoder: str = "vocoder_models/en/ljspeech/hifigan_v2"
    speaker: Optional[str] = None
    language: str = "en"
    emotion: str = "neutral"
    speed: float = 1.0
    device: str = "auto"


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    enabled: bool = True
    aggressiveness: int = 3
    frame_duration_ms: int = 30


@dataclass
class AudioInputConfig:
    """Audio input configuration"""
    device_index: Optional[int] = None
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    pause_threshold: float = 0.8
    phrase_timeout: float = 5.0


@dataclass
class AudioOutputConfig:
    """Audio output configuration"""
    device_index: Optional[int] = None
    volume: float = 0.8


@dataclass
class AudioConfig:
    """Audio system configuration"""
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format: str = "float32"
    vad: VADConfig = field(default_factory=VADConfig)
    input: AudioInputConfig = field(default_factory=AudioInputConfig)
    output: AudioOutputConfig = field(default_factory=AudioOutputConfig)


@dataclass
class UIFeaturesConfig:
    """UI features configuration"""
    voice_input: bool = True
    text_input: bool = True
    conversation_history: bool = True
    model_selection: bool = True
    settings_panel: bool = True


@dataclass
class UISecurityConfig:
    """UI security configuration"""
    cors_origins: list = field(default_factory=lambda: ["http://localhost:3000"])
    api_key: Optional[str] = None


@dataclass
class UIConfig:
    """User Interface configuration"""
    host: str = "localhost"
    port: int = 8080
    theme: str = "dark"
    title: str = "AskMe Voice Assistant"
    features: UIFeaturesConfig = field(default_factory=UIFeaturesConfig)
    security: UISecurityConfig = field(default_factory=UISecurityConfig)


@dataclass
class PrivacyConfig:
    """Privacy and data configuration"""
    store_conversations: bool = False
    conversation_retention_days: int = 7
    anonymize_logs: bool = True
    telemetry_enabled: bool = False


@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    size: int = 100
    ttl: int = 3600


@dataclass
class PerformanceConfig:
    """Performance settings"""
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    memory_optimization: bool = True
    cpu_optimization: bool = True
    cache: CacheConfig = field(default_factory=CacheConfig)


@dataclass
class DevelopmentConfig:
    """Development settings"""
    debug: bool = False
    hot_reload: bool = False
    profiling: bool = False
    mock_audio: bool = False


@dataclass
class ResponseFiltersConfig:
    """Response filtering configuration"""
    max_length: int = 1000
    filter_profanity: bool = True
    filter_personal_info: bool = True


@dataclass
class PluginsConfig:
    """Plugin system configuration"""
    enabled: list = field(default_factory=list)
    disabled: list = field(default_factory=list)
    search_paths: list = field(default_factory=lambda: ["./plugins"])


@dataclass
class AdvancedConfig:
    """Advanced configuration options"""
    system_prompt: str = """You are AskMe, a helpful offline voice assistant that prioritizes user privacy.
You operate entirely locally without internet connectivity.
Be conversational, helpful, and concise in your responses.
When you cannot perform internet-dependent tasks, suggest offline alternatives."""
    response_filters: ResponseFiltersConfig = field(default_factory=ResponseFiltersConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)


@dataclass
class QuantizationConfig:
    """Model quantization configuration"""
    enabled: bool = True
    method: str = "q4_0"


@dataclass
class MemoryConfig:
    """Memory management configuration"""
    low_memory_mode: bool = False
    swap_to_cpu: bool = True
    garbage_collection: bool = True


@dataclass
class ThreadingConfig:
    """Threading configuration"""
    max_workers: int = 4
    enable_parallel_processing: bool = True


@dataclass
class OptimizationsConfig:
    """Model and system optimizations"""
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    threading: ThreadingConfig = field(default_factory=ThreadingConfig)


class Config:
    """Main configuration class for AskMe Voice Assistant"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/config.yaml"
        self._config_data = {}
        self.load_config()
        self._setup_attributes()
    
    def load_config(self):
        """Load configuration from YAML file"""
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            # Try to load example config
            example_path = config_path.parent / "config.example.yaml"
            if example_path.exists():
                config_path = example_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _setup_attributes(self):
        """Setup configuration attributes from loaded data"""
        # Models configuration
        models_config = self._config_data.get('models', {})
        self.llm = ModelConfig(**models_config.get('llm', {}))
        self.asr = ASRConfig(**models_config.get('asr', {}))
        self.tts = TTSConfig(**models_config.get('tts', {}))
        
        # Audio configuration
        audio_config = self._config_data.get('audio', {})
        vad_config = VADConfig(**audio_config.get('vad', {}))
        input_config = AudioInputConfig(**audio_config.get('input', {}))
        output_config = AudioOutputConfig(**audio_config.get('output', {}))
        
        self.audio = AudioConfig(
            **{k: v for k, v in audio_config.items() if k not in ['vad', 'input', 'output']},
            vad=vad_config,
            input=input_config,
            output=output_config
        )
        
        # UI configuration
        ui_config = self._config_data.get('ui', {})
        features_config = UIFeaturesConfig(**ui_config.get('features', {}))
        security_config = UISecurityConfig(**ui_config.get('security', {}))
        
        self.ui = UIConfig(
            **{k: v for k, v in ui_config.items() if k not in ['features', 'security']},
            features=features_config,
            security=security_config
        )
        
        # Other configurations
        self.privacy = PrivacyConfig(**self._config_data.get('privacy', {}))
        
        # Performance with cache
        perf_config = self._config_data.get('performance', {})
        cache_config = CacheConfig(**perf_config.get('cache', {}))
        self.performance = PerformanceConfig(
            **{k: v for k, v in perf_config.items() if k != 'cache'},
            cache=cache_config
        )
        
        self.development = DevelopmentConfig(**self._config_data.get('development', {}))
        
        # Advanced configuration
        advanced_config = self._config_data.get('advanced', {})
        response_filters = ResponseFiltersConfig(**advanced_config.get('response_filters', {}))
        plugins = PluginsConfig(**advanced_config.get('plugins', {}))
        
        self.advanced = AdvancedConfig(
            **{k: v for k, v in advanced_config.items() if k not in ['response_filters', 'plugins']},
            response_filters=response_filters,
            plugins=plugins
        )
        
        # Optimizations configuration
        opt_config = self._config_data.get('optimizations', {})
        quant_config = QuantizationConfig(**opt_config.get('quantization', {}))
        memory_config = MemoryConfig(**opt_config.get('memory', {}))
        threading_config = ThreadingConfig(**opt_config.get('threading', {}))
        
        self.optimizations = OptimizationsConfig(
            quantization=quant_config,
            memory=memory_config,
            threading=threading_config
        )
        
        # Logging configuration
        logging_config = self._config_data.get('logging', {})
        self.log_level = logging_config.get('level', 'INFO')
        self.log_file = logging_config.get('file', 'logs/askme.log')
        self.log_max_size = logging_config.get('max_file_size', '10MB')
        self.log_backup_count = logging_config.get('backup_count', 5)
        self.log_format = logging_config.get('format', 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def get_model_path(self) -> str:
        """Get the full path to the LLM model file"""
        return os.path.join(self.llm.path, self.llm.model_file)
    
    def validate(self) -> list:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate model path
        model_path = Path(self.get_model_path())
        if not model_path.exists():
            issues.append(f"LLM model file not found: {model_path}")
        
        # Validate port range
        if not (1024 <= self.ui.port <= 65535):
            issues.append(f"Invalid port number: {self.ui.port}")
        
        # Validate audio settings
        if self.audio.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            issues.append(f"Unsupported sample rate: {self.audio.sample_rate}")
        
        # Validate temperature range
        if not (0.0 <= self.llm.temperature <= 2.0):
            issues.append(f"Invalid temperature value: {self.llm.temperature}")
        
        return issues
    
    def save(self, output_path: Optional[str] = None):
        """Save current configuration to file"""
        output_path = output_path or self.config_path
        
        # Convert dataclasses back to dictionaries
        config_dict = {
            'models': {
                'llm': self.llm.__dict__,
                'asr': self.asr.__dict__,
                'tts': self.tts.__dict__
            },
            'audio': {
                **{k: v for k, v in self.audio.__dict__.items() if k not in ['vad', 'input', 'output']},
                'vad': self.audio.vad.__dict__,
                'input': self.audio.input.__dict__,
                'output': self.audio.output.__dict__
            },
            'ui': {
                **{k: v for k, v in self.ui.__dict__.items() if k not in ['features', 'security']},
                'features': self.ui.features.__dict__,
                'security': self.ui.security.__dict__
            },
            'privacy': self.privacy.__dict__,
            'performance': {
                **{k: v for k, v in self.performance.__dict__.items() if k != 'cache'},
                'cache': self.performance.cache.__dict__
            },
            'development': self.development.__dict__,
            'advanced': {
                **{k: v for k, v in self.advanced.__dict__.items() if k not in ['response_filters', 'plugins']},
                'response_filters': self.advanced.response_filters.__dict__,
                'plugins': self.advanced.plugins.__dict__
            },
            'optimizations': {
                'quantization': self.optimizations.quantization.__dict__,
                'memory': self.optimizations.memory.__dict__,
                'threading': self.optimizations.threading.__dict__
            },
            'logging': {
                'level': self.log_level,
                'file': self.log_file,
                'max_file_size': self.log_max_size,
                'backup_count': self.log_backup_count,
                'format': self.log_format
            }
        }
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to YAML file
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"AskMe Configuration (LLM: {self.llm.name}, ASR: {self.asr.model}, TTS: {self.tts.model})"
