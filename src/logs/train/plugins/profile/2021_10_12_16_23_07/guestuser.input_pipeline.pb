	l�`A��@l�`A��@!l�`A��@	���Z�q?���Z�q?!���Z�q?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$l�`A��@��W���?A�Ue�U��@YǸ��ܰ?*	ffffffV@2F
Iterator::Model�7ӅX�?!m۶m�CG@)ᶶ�T�?1n۶m��>@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat���6��?!m۶m��:@)5D�o�?1�m۶ms8@:Preprocessing2S
Iterator::Model::ParallelMap�q�j���?!ܶm۶M/@)�q�j���?1ܶm۶M/@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�	��.�?!۶m۶e2@)	����?1ܶm۶�)@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip�=^H���?!�$I�$�J@)M�St$w?1�$I�$9@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice#��t?!�m۶m�@)#��t?1�m۶m�@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor�5�;N�a?!�m۶mk@)�5�;N�a?1�m۶mk@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap��I����?!�$I�$I4@)���*ø[?1n۶m�6�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��W���?��W���?!��W���?      ��!       "      ��!       *      ��!       2	�Ue�U��@�Ue�U��@!�Ue�U��@:      ��!       B      ��!       J	Ǹ��ܰ?Ǹ��ܰ?!Ǹ��ܰ?R      ��!       Z	Ǹ��ܰ?Ǹ��ܰ?!Ǹ��ܰ?JCPU_ONLY