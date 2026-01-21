##export TORCH_LOGS="cudagraph_static_inputs,cudagraphs,graph_breaks,recompiles"
#export TORCH_LOGS="cudagraphs,graph_breaks,recompiles"
#export TORCHDYNAMO_VERBOSE=1
#export CUDA_LAUNCH_BLOCKING=1
#export PYTHONFAULTHANDLER=1
OMP_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=4 numactl -C 32-63 -l python cosyvoice3.py --text "hello world~ My name is cosy voice version 3.0" --stream #--profile ../profile_cuda_stream.json --iteration 5

