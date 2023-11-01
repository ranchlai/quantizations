import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Update JSON configuration files")
    parser.add_argument("--config_file", help="Path to the config.json file")
    parser.add_argument("--quantize_config_file", help="Path to the quantize_config.json file")
    
    args = parser.parse_args()

    # Load config.json
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Add quantize_method=gptq to quantize_config.json
    with open(args.quantize_config_file, 'r') as f:
        quantize_config = json.load(f)
        
    quantize_config['quant_method'] = 'gptq'

    # Combine quantize_config to config
    config.update({'quantization_config': quantize_config})


    # Save the updated config.json
    with open(args.config_file, 'w') as f:
        json.dump(config, f, indent=4)
        
    print("Updated config.json with quantization_config.json")
    print("config.json:", config)

if __name__ == "__main__":
    main()
