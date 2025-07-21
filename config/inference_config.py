class InferenceConfig:
    # Model configuration
    MODEL_DIM = 512
    N_ENC = 6
    N_DEC = 6
    MAX_SEQ_LEN = 100
    VOCAB_SIZE = 10201
    
    # Device configuration
    MAX_DEVICES = 8
    DEFAULT_DEVICES = 4
    
    # Splitting strategies
    SPLITTING_STRATEGIES = {
        'static_even': 'Divide layers evenly across devices',
        'static_weighted': 'Weight division by device capabilities',
        'adaptive': 'Use RL policy for adaptive splitting',
        'bandwidth_aware': 'Split based on network bandwidth'
    }
    
    # Performance thresholds
    MIN_BANDWIDTH_MBPS = 10
    MAX_LATENCY_MS = 200
    MIN_DEVICE_CAPABILITY = 0.2

class DeviceProfiles:
    PROFILES = {
        'mobile': {
            'compute_power': 0.3,
            'memory': 0.4,
            'power_budget': 0.4,
            'typical_bandwidth': 50
        },
        'edge': {
            'compute_power': 0.5,
            'memory': 0.6,
            'power_budget': 0.7,
            'typical_bandwidth': 100
        },
        'server': {
            'compute_power': 1.0,
            'memory': 1.0,
            'power_budget': 0.9,
            'typical_bandwidth': 1000
        }
    }
