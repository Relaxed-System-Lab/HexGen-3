# sparse flops == dense flops // 2
# Prices from vast.ai ($/hr, P25 = 25th percentile) as of Nov 2025
hardware_params = {
    "NVDA:RTX3090": {
        "bandwidth": 936e9,
        "FP16": 71e12,
        "INT8": 71e12,
        "onchip_buffer": 18432e3,
        "vmemory": 24e9,  # 24GB in bytes
        "price_per_hour": 0.13,
    },
    "NVDA:A6000": {
        "bandwidth": 1041e9,
        "FP16": 419.892e12 / 2,
        "INT8": 419.892e12,
        "onchip_buffer": 21504e3,
        "vmemory": 48e9,  # 48GB in bytes
        "price_per_hour": 0.39,
    },
    "NVDA:A100": {
        "bandwidth": 1555e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "vmemory": 40e9,  # 40GB in bytes
        "price_per_hour": 0.65,  # PCIE version
    },  # use 40G data
    "NVDA:A100_40G:SXM": {
        "bandwidth": 2039e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "vmemory": 40e9,
        "price_per_hour": 0.68,
    },
    "NVDA:A100_80G:SXM": {
        "bandwidth": 2039e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "vmemory": 80e9,  # 80GB in bytes
        "price_per_hour": 0.68,
    },
    "NVDA:A800_80G:SXM": {
        "bandwidth": 2039e9,
        "FP16": 312e12,
        "INT8": 624e12,
        "onchip_buffer": 27648e3,
        "vmemory": 80e9,  # 80GB in bytes
        "price_per_hour": 0.68,  # Assumed same as A100 SXM
    },
    "NVDA:A40S": {
        "bandwidth": 696e9,
        "FP16": 149.7e12,
        "INT8": 299.3e12,
        "onchip_buffer": 21504e3,
        "vmemory": 48e9,  # 48GB in bytes
        "price_per_hour": 0.30,  # Estimated
    },
    "NVDA:H100:SXM": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
        "vmemory": 80e9,
        "price_per_hour": 1.56,
    },
    "NVDA:H100:PCIe": {
        "bandwidth": 2048e9,
        "FP16": 1513e12 / 2,
        "INT8": 3026e12 / 2,
        "onchip_buffer": 29184e3,
        "vmemory": 80e9,
        "price_per_hour": 1.20,  # Estimated lower than SXM
    },
    "NVDA:H20": {
        "bandwidth": 4096e9,
        "FP16": 148e12,
        "INT8": 296e12,
        "onchip_buffer": 34560e3,
        "vmemory": 960e8,
        "price_per_hour": 2
    },
    "NVDA:H800": {
        "bandwidth": 3430e9,  
        "FP16": 989e12,             
        "INT8": 989e12 * 2,   
        "onchip_buffer": 30720e3,    
        "vmemory": 800e8,               
        "price_per_hour": 5.28,
    },
    "NVDA:L40S": {
        "bandwidth": 864e9,
        "FP16": 181e12,
        "INT8": 362e12,
        "onchip_buffer": 36352e3,
        "vmemory": 48e9,
        "price_per_hour": 0.44,
    },
    "NVDA:GH200": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
        "vmemory": 80e9,
        "price_per_hour": 1.56,
    },
}