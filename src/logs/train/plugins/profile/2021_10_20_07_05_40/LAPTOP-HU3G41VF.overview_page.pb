�	{��!�n@{��!�n@!{��!�n@	��Z��?��Z��?!��Z��?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:{��!�n@]�z�ε?A����_l@Y�c�g^�?rEagerKernelExecute 0*	��(\�rJ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatc+hZbe�?!zT���B@)tzލ��?1�<��/,@@:Preprocessing2U
Iterator::Model::ParallelMapV2�8K�r�?!���ɫ1@)�8K�r�?1���ɫ1@:Preprocessing2F
Iterator::Model�$y��Ñ?!C�^�Cf@@)�<��?1�OB��/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���+҃?!g�<��K2@)F(���%v?1n��q$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicec�#�w~q?!`�g/#& @)c�#�w~q?1`�g/#& @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���3�?!ߎP%��P@),����n?1��茛@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr�Md�g?!]�-�=@)r�Md�g?1]�-�=@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���0B�?!�i�;�d6@)�,D��a?1��^c@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��Z��?IX)��S�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	]�z�ε?]�z�ε?!]�z�ε?      ��!       "      ��!       *      ��!       2	����_l@����_l@!����_l@:      ��!       B      ��!       J	�c�g^�?�c�g^�?!�c�g^�?R      ��!       Z	�c�g^�?�c�g^�?!�c�g^�?b      ��!       JCPU_ONLYY��Z��?b qX)��S�X@Y      Y@qkZR\�|�?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 