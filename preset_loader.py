"""
Preset Loader for nanoGPT-Ockham Plugin System

Loads plugin configurations from YAML files and creates plugin chains.
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from plugin_system import PluginHost, LearningPlugin
from plugins import (
    CompressorPlugin,
    EQPlugin,
    LimiterPlugin,
    SaturationPlugin,
    OckhamGatePlugin,
)


# Plugin registry: maps type names to classes
PLUGIN_REGISTRY = {
    'CompressorPlugin': CompressorPlugin,
    'EQPlugin': EQPlugin,
    'LimiterPlugin': LimiterPlugin,
    'SaturationPlugin': SaturationPlugin,
    'OckhamGatePlugin': OckhamGatePlugin,
}


class PresetLoader:
    """
    Loads and manages plugin presets from YAML files.
    
    Preset files define:
    - Plugin chain (types and parameters)
    - Base hyperparameters
    - Metadata (name, description)
    """
    
    def __init__(self, presets_dir: str = "presets"):
        self.presets_dir = presets_dir
        self.loaded_presets = {}
    
    def load_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Load a preset from YAML file.
        
        Args:
            preset_name: Name of preset file (without .yaml extension)
        
        Returns:
            Dictionary containing preset configuration
        """
        # Check if already loaded
        if preset_name in self.loaded_presets:
            return self.loaded_presets[preset_name]
        
        # Load from file
        preset_path = os.path.join(self.presets_dir, f"{preset_name}.yaml")
        
        if not os.path.exists(preset_path):
            raise FileNotFoundError(f"Preset not found: {preset_path}")
        
        with open(preset_path, 'r') as f:
            preset = yaml.safe_load(f)
        
        # Cache
        self.loaded_presets[preset_name] = preset
        
        return preset
    
    def create_plugin_host(self, preset_name: str) -> PluginHost:
        """
        Create a PluginHost from a preset.
        
        Args:
            preset_name: Name of preset to load
        
        Returns:
            PluginHost with configured plugins
        """
        preset = self.load_preset(preset_name)
        
        plugins = []
        for plugin_config in preset.get('plugins', []):
            plugin_type = plugin_config['type']
            plugin_params = plugin_config.get('params', {})
            
            # Look up plugin class
            plugin_class = PLUGIN_REGISTRY.get(plugin_type)
            if plugin_class is None:
                raise ValueError(f"Unknown plugin type: {plugin_type}")
            
            # Create plugin instance
            plugin = plugin_class(**plugin_params)
            plugins.append(plugin)
        
        return PluginHost(plugins=plugins)
    
    def get_hyperparameters(self, preset_name: str) -> Dict[str, Any]:
        """
        Get base hyperparameters from a preset.
        
        Args:
            preset_name: Name of preset to load
        
        Returns:
            Dictionary of hyperparameters
        """
        preset = self.load_preset(preset_name)
        return preset.get('hyperparameters', {})
    
    def get_preset_info(self, preset_name: str) -> Dict[str, str]:
        """
        Get metadata about a preset.
        
        Args:
            preset_name: Name of preset to load
        
        Returns:
            Dictionary with 'name' and 'description'
        """
        preset = self.load_preset(preset_name)
        return {
            'name': preset.get('name', preset_name),
            'description': preset.get('description', ''),
        }
    
    def list_presets(self) -> List[str]:
        """
        List all available presets in the presets directory.
        
        Returns:
            List of preset names (without .yaml extension)
        """
        if not os.path.exists(self.presets_dir):
            return []
        
        presets = []
        for filename in os.listdir(self.presets_dir):
            if filename.endswith('.yaml'):
                preset_name = filename[:-5]  # Remove .yaml
                presets.append(preset_name)
        
        return sorted(presets)
    
    def print_preset_summary(self, preset_name: str) -> None:
        """Print a human-readable summary of a preset."""
        preset = self.load_preset(preset_name)
        info = self.get_preset_info(preset_name)
        
        print("=" * 80)
        print(f"PRESET: {info['name']}")
        print("=" * 80)
        print(f"\nDescription: {info['description']}")
        
        print("\nPlugins:")
        for i, plugin_config in enumerate(preset.get('plugins', []), 1):
            plugin_type = plugin_config['type']
            plugin_params = plugin_config.get('params', {})
            print(f"  {i}. {plugin_type}")
            for key, value in plugin_params.items():
                print(f"     - {key}: {value}")
        
        print("\nBase Hyperparameters:")
        hparams = preset.get('hyperparameters', {})
        for key, value in hparams.items():
            print(f"  - {key}: {value}")
        
        print("=" * 80)


def load_preset_simple(preset_name: str, presets_dir: str = "presets") -> PluginHost:
    """
    Simple helper function to load a preset and create a PluginHost.
    
    Args:
        preset_name: Name of preset to load
        presets_dir: Directory containing preset files
    
    Returns:
        PluginHost with configured plugins
    
    Example:
        >>> host = load_preset_simple("balanced")
        >>> state = host.process_batch_start(state)
    """
    loader = PresetLoader(presets_dir=presets_dir)
    return loader.create_plugin_host(preset_name)


if __name__ == "__main__":
    from plugin_system import TrainingState
    
    print("=" * 80)
    print("PRESET LOADER DEMONSTRATION")
    print("=" * 80)
    
    # Create loader
    loader = PresetLoader(presets_dir="presets")
    
    # List available presets
    print("\nAvailable presets:")
    for preset_name in loader.list_presets():
        info = loader.get_preset_info(preset_name)
        print(f"  - {preset_name}: {info['description']}")
    
    print("\n" + "-" * 80)
    
    # Load and display each preset
    for preset_name in loader.list_presets():
        print()
        loader.print_preset_summary(preset_name)
        print()
    
    print("-" * 80)
    
    # Test loading a preset
    print("\nTesting 'balanced' preset...")
    host = loader.create_plugin_host("balanced")
    hparams = loader.get_hyperparameters("balanced")
    
    print(f"Created host with {len(host)} plugins:")
    for plugin in host.plugins:
        print(f"  - {plugin.name} ({plugin.__class__.__name__})")
    
    print(f"\nBase hyperparameters: {hparams}")
    
    # Simulate one batch
    print("\nSimulating one batch...")
    state = TrainingState(
        iter_num=0,
        learning_rate=hparams['learning_rate'],
        lambda_ockham=hparams['lambda_ockham'],
        surprise_threshold=hparams['surprise_threshold'],
        task_loss=2.5,
        complexity_cost=0.08,
        grad_norm=0.5,
    )
    
    print(f"Before: lr={state.learning_rate:.6f}, λ={state.lambda_ockham:.4f}")
    state = host.process_batch_start(state)
    print(f"After:  lr={state.learning_rate:.6f}, λ={state.lambda_ockham:.4f}")
    
    print("\n✓ Demo complete!")
