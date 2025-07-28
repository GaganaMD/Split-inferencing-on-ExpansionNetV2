import torch
import time
from typing import Dict, List, Any
from ..models.expansionnet_inference import SplitConfiguration

class InferenceCoordinator:
    """Coordinates multi-device inference execution"""
    
    def __init__(self):
        self.communication_overhead = 0.001  # Base communication overhead per MB
    
    def execute_pipeline(self, model, split_config: SplitConfiguration,
                        images: torch.Tensor, captions: torch.Tensor,
                        device_capabilities: Dict, network_conditions: tuple) -> Dict[str, Any]:
        """Execute the inference pipeline across multiple devices"""
        
        device_results = {}
        communication_times = {}
        total_data_transferred = 0
        
        # Prepare initial data
        pipeline_data = {
            'images': images,
            'captions': captions
        }
        
        try:
            # Execute pipeline sequentially
            for i, assignment in enumerate(split_config.device_assignments):
                device_id = assignment.device_id
                
                # Simulate device processing
                device_start_time = time.time()
                
                with torch.no_grad():
                    device_output = model.forward_device_segment(
                        device_id, pipeline_data, split_config
                    )
                
                device_processing_time = time.time() - device_start_time
                
                # Store results
                device_results[device_id] = {
                    'processing_time': device_processing_time,
                    'output_keys': list(device_output.keys()),
                    'assignment': assignment
                }
                
                # Simulate communication to next device
                if i < len(split_config.device_assignments) - 1:
                    next_device = split_config.device_assignments[i + 1].device_id
                    
                    # Estimate data transfer
                    transfer_data = self._extract_transfer_data(device_output)
                    data_size = self._estimate_data_size(transfer_data)
                    
                    # Simulate communication delay
                    comm_time = self._simulate_communication_delay(
                        data_size, device_id, next_device, network_conditions
                    )
                    
                    communication_times[(device_id, next_device)] = comm_time
                    total_data_transferred += data_size
                
                # Update pipeline data for next device
                pipeline_data.update(device_output)
            
            # Extract final results
            final_logits = pipeline_data.get('final_logits')
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(
                device_results, communication_times, total_data_transferred,
                final_logits, captions
            )
            
            return {
                'final_logits': final_logits,
                'device_results': device_results,
                'communication_times': communication_times,
                'total_data_transferred': total_data_transferred,
                'metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            # Fallback to single device
            return self._fallback_single_device(model, images, captions)
    
    def _extract_transfer_data(self, device_output: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Extract data that needs to be transferred to next device"""
        transfer_keys = ['encoder_output', 'encoder_complete', 'decoder_output', 'visual_features']
        transfer_data = {}
        
        for key in transfer_keys:
            if key in device_output and isinstance(device_output[key], torch.Tensor):
                transfer_data[key] = device_output[key]
        
        return transfer_data
    
    def _estimate_data_size(self, data: Dict[str, torch.Tensor]) -> float:
        """Estimate data size in MB"""
        total_bytes = 0
        
        for tensor in data.values():
            if isinstance(tensor, torch.Tensor):
                total_bytes += tensor.numel() * 4  # Assume float32
        
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _simulate_communication_delay(self, data_size_mb: float, from_device: int, 
                                     to_device: int, network_conditions: tuple) -> float:
        """Simulate communication delay between devices"""
        bandwidths, latencies = network_conditions
        
        # Get network conditions for this link
        link_idx = min(from_device, len(bandwidths) - 1)
        bandwidth_mbps = bandwidths[link_idx] if bandwidths else 100
        latency_ms = latencies[link_idx] if latencies else 20
        
        # Calculate transfer time
        transfer_time = (data_size_mb * 8) / bandwidth_mbps  # Convert MB to Mb, divide by Mbps
        total_delay = transfer_time + (latency_ms / 1000.0)  # Add latency
        
        # Add communication overhead
        overhead = data_size_mb * self.communication_overhead
        
        return total_delay + overhead
    
    def _calculate_performance_metrics(self, device_results: Dict, 
                                     communication_times: Dict, 
                                     total_data_transferred: float,
                                     final_logits: torch.Tensor,
                                     captions: torch.Tensor) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        # Processing time metrics
        processing_times = [result['processing_time'] for result in device_results.values()]
        total_processing_time = sum(processing_times)
        max_processing_time = max(processing_times) if processing_times else 0
        
        # Communication metrics
        total_communication_time = sum(communication_times.values()) if communication_times else 0
        
        # Load balancing
        if len(processing_times) > 1:
            time_std = torch.std(torch.tensor(processing_times)).item()
            load_balance = 1.0 - (time_std / max_processing_time) if max_processing_time > 0 else 1.0
        else:
            load_balance = 1.0
        
        # Accuracy (simplified)
        accuracy = 0.0
        if final_logits is not None and captions is not None:
            predictions = torch.argmax(final_logits, dim=-1)
            correct = (predictions == captions).float().mean().item()
            accuracy = correct
        
        # Efficiency metrics
        total_time = max_processing_time + total_communication_time
        parallelization_efficiency = total_processing_time / total_time if total_time > 0 else 0
        communication_efficiency = 1.0 - (total_communication_time / total_time) if total_time > 0 else 1.0
        
        return {
            'device_count': len(device_results),
            'total_processing_time': total_processing_time,
            'max_processing_time': max_processing_time,
            'total_communication_time': total_communication_time,
            'total_data_transferred_mb': total_data_transferred,
            'load_balance_score': load_balance,
            'accuracy': accuracy,
            'parallelization_efficiency': parallelization_efficiency,
            'communication_efficiency': communication_efficiency,
            'total_pipeline_time': total_time
        }
    
    def execute_pipeline(self, model, split_config: SplitConfiguration, images: torch.Tensor, captions: torch.Tensor,
                         device_capabilities: Dict, network_conditions: tuple,
                    debug_mode: bool = True) -> Dict[str, Any]:
        """Execute pipeline with enhanced debugging"""
        
        device_results = {}
        communication_times = {}
        total_data_transferred = 0
        
        # Debug: Track tensor flow
        tensor_flow_log = []
        
        # Prepare initial data
        pipeline_data = {
            'images': images,
            'captions': captions
        }
        
        if debug_mode:
            print(f" DEBUG: Initial pipeline data keys: {list(pipeline_data.keys())}")
            print(f" DEBUG: Images shape: {images.shape}")
            print(f" DEBUG: Captions shape: {captions.shape}")
        
        try:
            # Execute pipeline sequentially with debugging
            for i, assignment in enumerate(split_config.device_assignments):
                device_id = assignment.device_id
                
                if debug_mode:
                    print(f"\n📱 Device {device_id} (layers {assignment.layer_start}-{assignment.layer_end}):")
                    print(f"   Input keys: {list(pipeline_data.keys())}")
                    for key, tensor in pipeline_data.items():
                        if isinstance(tensor, torch.Tensor):
                            print(f"   {key} shape: {tensor.shape}")
                
                # Device processing with debugging
                device_start_time = time.time()
                
                with torch.no_grad():
                    device_output = model.forward_device_segment(
                        device_id, pipeline_data, split_config
                    )
                
                device_processing_time = time.time() - device_start_time
                
                # Debug: Log device output
                if debug_mode:
                    print(f"   Output keys: {list(device_output.keys())}")
                    for key, tensor in device_output.items():
                        if isinstance(tensor, torch.Tensor):
                            print(f"   {key} shape: {tensor.shape}")
                    
                    # Critical check for encoder completion
                    if 'encoder_complete' in device_output:
                        print(f"   ✅ ENCODER COMPLETE found on device {device_id}")
                        print(f"   Shape: {device_output['encoder_complete'].shape}")
                    else:
                        print(f"   ⚠️  No encoder_complete from device {device_id}")
                
                # Store results with debugging info
                device_results[device_id] = {
                    'processing_time': device_processing_time,
                    'output_keys': list(device_output.keys()),
                    'assignment': assignment,
                    'debug_info': {
                        'had_encoder_complete': 'encoder_complete' in device_output,
                        'had_decoder_output': 'decoder_output' in device_output,
                        'had_final_logits': 'final_logits' in device_output
                    }
                }
                
                # Log tensor flow for debugging
                tensor_flow_log.append({
                    'device_id': device_id,
                    'input_keys': list(pipeline_data.keys()),
                    'output_keys': list(device_output.keys()),
                    'encoder_complete_present': 'encoder_complete' in device_output
                })
                
                # Communication simulation and debugging
                if i < len(split_config.device_assignments) - 1:
                    next_device = split_config.device_assignments[i + 1].device_id
                    
                    # Extract transfer data
                    transfer_data = self._extract_transfer_data(device_output)
                    data_size = self._estimate_data_size(transfer_data)
                    
                    if debug_mode:
                        print(f"   📡 Transferring to device {next_device}: {list(transfer_data.keys())}")
                        print(f"   Data size: {data_size:.2f} MB")
                    
                    # Simulate communication delay
                    comm_time = self._simulate_communication_delay(
                        data_size, device_id, next_device, network_conditions
                    )
                    
                    communication_times[(device_id, next_device)] = comm_time
                    total_data_transferred += data_size
                
                # Update pipeline data for next device
                pipeline_data.update(device_output)
                
                if debug_mode:
                    print(f"   Updated pipeline data keys: {list(pipeline_data.keys())}")
            
            # Final debugging check
            final_logits = pipeline_data.get('final_logits')
            if debug_mode:
                print(f"\n Final Results:")
                print(f"   Final logits present: {final_logits is not None}")
                if final_logits is not None:
                    print(f"   Final logits shape: {final_logits.shape}")
                print(f"   Final pipeline keys: {list(pipeline_data.keys())}")
            
            # Calculate metrics with debugging
            metrics = self._calculate_performance_metrics(
                device_results, communication_times, total_data_transferred,
                final_logits, captions
            )
            
            # Add debugging information to results
            return {
                'final_logits': final_logits,
                'device_results': device_results,
                'communication_times': communication_times,
                'total_data_transferred': total_data_transferred,
                'metrics': metrics,
                'success': True,
                'debug_info': {
                    'tensor_flow_log': tensor_flow_log,
                    'encoder_complete_devices': [
                        log['device_id'] for log in tensor_flow_log 
                        if log['encoder_complete_present']
                    ]
                }
            }
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_single_device(model, images, captions)

    
    def _fallback_single_device(self, model, images: torch.Tensor, 
                               captions: torch.Tensor) -> Dict[str, Any]:
        """Fallback to single device inference"""
        print("Falling back to single device inference...")
        
        start_time = time.time()
        
        with torch.no_grad():
            final_logits = model(images, captions)
        
        processing_time = time.time() - start_time
        
        # Basic accuracy
        accuracy = 0.0
        if final_logits is not None and captions is not None:
            predictions = torch.argmax(final_logits, dim=-1)
            correct = (predictions == captions).float().mean().item()
            accuracy = correct
        
        return {
            'final_logits': final_logits,
            'device_results': {0: {'processing_time': processing_time}},
            'communication_times': {},
            'total_data_transferred': 0,
            'metrics': {
                'device_count': 1,
                'total_processing_time': processing_time,
                'accuracy': accuracy,
                'load_balance_score': 1.0,
                'parallelization_efficiency': 1.0,
                'communication_efficiency': 1.0
            },
            'success': True,
            'fallback': True
        }
