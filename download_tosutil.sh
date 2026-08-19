./tosutil config \
    -i AKLTZjdkNDdkYmFlYThmNGE3ODk3ZjY3ZWNmYjRkZGM3ODM \
    -k TmpoaU9XSmlNalUyWkdKbE5HRTRPVGd3WVdOaE5HTTNOR00xTm1Sa09HUQ== \
    -e tos-cn-beijing.volces.com \
    -re cn-beijing

./tosutil cp tos://simpleai-vla/mjm-data/pi05-data/checkpoints/pi05_place_remote_new_bs512/ /home/simpleai/pi05-deploy/checkpoints/pi05_place_remote_new_bs512_9w/  -r -u

./tosutil cp tos://simpleai-vla/mjm-data/pi05-data/checkpoints/pi05_task485_bs256_JAX_8gpu_5w/ /home/simpleai/pi05-deploy/checkpoints/pi05_task485_bs256_JAX_8gpu_5w/  -r -u

./tosutil cp tos://simpleai-vla/mjm-data/pi05-data/checkpoints/pi05_task483_ds3_bs256_JAX_16gpu_10hz_3w/ /home/simpleai/pi05-deploy/checkpoints/pi05_task483_ds3_bs256_JAX_16gpu_10hz_3w/  -r -u
