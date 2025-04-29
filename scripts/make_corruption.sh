# python ./make_corruptions/make_c_video.py --corruption 'all' --severity 5 --data_path 'data_path/Kinetics50/image_mulframe_val256_k=50' --save_path 'data_path/Kinetics50/image_mulframe_val256_k=50-C'
# corr=('frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression')
# for c in ${corr[@]}
# do
#     python ./make_corruptions/make_c_video.py --corruption $c --severity 5 --data_path 'data_path/Kinetics50/image_mulframe_val256_k=50' --save_path 'data_path/Kinetics50/image_mulframe_val256_k=50-C'
# done
# python ./make_corruptions/make_c_audio.py --corruption 'all' --severity 5 --data_path 'data_path/Kinetics50/audio_val256_k=50' --save_path 'data_path/Kinetics50/audio_val256_k=50-C' --weather_path 'data_path/NoisyAudios'
# python ./make_corruptions/make_c_audio.py --corruption 'gaussian_noise' --severity 5 --data_path 'data_path/Kinetics50/audio_val256_k=50' --save_path 'data_path/Kinetics50/audio_val256_k=50-C' --weather_path 'data_path/NoisyAudios'
# python ./make_corruptions/make_c_video.py --corruption 'all' --severity 5 --data_path 'data_path/VGGSound/image_mulframe_test' --save_path 'data_path/VGGSound/image_mulframe_test-C'
# python ./make_corruptions/make_c_audio.py --corruption 'all' --severity 5 --data_path 'data_path/VGGSound/audio_test' --save_path 'data_path/VGGSound/audio_test-C' --weather_path 'data_path/NoisyAudios'
severities=(1 2 3 4)
for severity in ${severities[@]}
do
    python ./make_corruptions/make_c_video.py --corruption 'all' --severity $severity --data_path 'data_path/Kinetics50/image_mulframe_val256_k=50' --save_path 'data_path/Kinetics50/image_mulframe_val256_k=50-C'
done