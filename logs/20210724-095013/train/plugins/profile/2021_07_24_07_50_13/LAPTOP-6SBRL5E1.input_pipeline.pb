	??e?c]????e?c]??!??e?c]??	?G?A?@?G?A?@!?G?A?@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??e?c]??Έ?????A???3???YS?!?uq??rEagerKernelExecute 0*	fffff&U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatp_?Q??!???`>@)A??ǘ???1Y???=:@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??Pk?w??!Dٿ?%n@@)U???N@??1??8??86@:Preprocessing2U
Iterator::Model::ParallelMapV2?5?;Nё?!?)͋??4@)?5?;Nё?1?)͋??4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice;?O??n??!?????F%@);?O??n??1?????F%@:Preprocessing2F
Iterator::Model????????!?>??=@)ŏ1w-!?1]*????!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip2??%䃮?!D?U>??Q@)?HP?x?1????s?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensory?&1?l?!?A5h?@)y?&1?l?1?A5h?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?G?A?@I?{/???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Έ?????Έ?????!Έ?????      ??!       "      ??!       *      ??!       2	???3??????3???!???3???:      ??!       B      ??!       J	S?!?uq??S?!?uq??!S?!?uq??R      ??!       Z	S?!?uq??S?!?uq??!S?!?uq??b      ??!       JCPU_ONLYY?G?A?@b q?{/???W@