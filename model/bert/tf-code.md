# BERT TensorFlow源码

## 预处理

**主函数**

```python
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # 定义分词
    tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    # 获取输入文件
  	input_files = []
  	for input_pattern in FLAGS.input_file.split(","):
    	input_files.extend(tf.gfile.Glob(input_pattern))
  	tf.logging.info("*** Reading from input files ***")
  	for input_file in input_files:
    	tf.logging.info("  %s", input_file)
        
    # 生成训练样本
  	rng = random.Random(FLAGS.random_seed)
  	instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      	FLAGS.short_seq_prob, FLAGS.masked_lm_prob,
        FLAGS.max_predictions_per_seq, rng)
    
    # 输出
  	output_files = FLAGS.output_file.split(",")
  	tf.logging.info("*** Writing to output files ***")
  	for output_file in output_files:
    	tf.logging.info("  %s", output_file)
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)
```

**生成训练样本**

```python
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    all_documents = [[]]
    for input_file in input_files:
    	with tf.gfile.GFile(input_file, "r") as reader:
            l = 0
            while True:
                # 转码
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()
                # 空行划分documents
                if not line:
                    all_documents.append([])
                # 分词
        		tokens = tokenizer.tokenize(line)
        		if tokens:
          			all_documents[-1].append(tokens)

	all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    
    # 生成词典
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
    	for document_index in range(len(all_documents)):
            # 原子操作
            instances.extend(
                create_instances_from_document(all_documents, document_index,
                	max_seq_length, short_seq_prob, masked_lm_prob,
                    max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances
```

**document原子操作**

```python
def create_instances_from_document(all_documents, document_index,
                                   max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq,
                                   vocab_words, rng):
    document = all_documents[document_index]
    # 减去[CLS]、[SEP]、[SEP]
    max_num_tokens = max_seq_length - 3
    target_seq_length = max_num_tokens
    # 概率变成短句
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        # 逐句添加进chunk
        segment = document[i]
        current_chunk.append(segment)
        # 句子长度
        current_length += len(segment)
    	if i == len(document) - 1 or current_length >= target_seq_length:
        	if current_chunk:
        		# 随机选定sentence A的末尾位置
        		a_end = 1
        		if len(current_chunk) >= 2:
          			a_end = rng.randint(1, len(current_chunk) - 1)

        		tokens_a = []
        		for j in range(a_end):
          			tokens_a.extend(current_chunk[j])

        		tokens_b = []
        		is_random_next = False
                # sentence B不是下一句
        		if len(current_chunk) == 1 or rng.random() < 0.5:
          			is_random_next = True
                    # sentence B最大长度
          			target_b_length = target_seq_length - len(tokens_a)
                    # 在不同document中采样
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

          			random_document = all_documents[random_document_index]
          			random_start = rng.randint(0, len(random_document) - 1)
                    # 添加sentence B直到最长
          			for j in range(random_start, len(random_document)):
            			tokens_b.extend(random_document[j])
            			if len(tokens_b) >= target_b_length:
              				break
                    # 如果B不是A下一句，则A的下一句并未被采样到，放回
          			num_unused_segments = len(current_chunk) - a_end
          			i -= num_unused_segments
        		# sentence B是下一句
        		else:
                    is_random_next = False
                    # 从a_end开始顺序添加
          			for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                # 截短
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
                
                tokens = []
        		segment_ids = []
                # 添加占位符和句子信息
        		tokens.append("[CLS]")
        		segment_ids.append(0)
        		for token in tokens_a:
          			tokens.append(token)
          			segment_ids.append(0)
                tokens.append("[SEP]")
        		segment_ids.append(0)
        		for token in tokens_b:
                    tokens.append(token)
          			segment_ids.append(1)
        		tokens.append("[SEP]")
        		segment_ids.append(1)
                
                # 打mask
                (tokens, masked_lm_positions, masked_lm_labels) =
                	create_masked_lm_predictions(tokens, masked_lm_prob,                                                                  max_predictions_per_seq,
                                                 vocab_words, rng)
        		instance = TrainingInstance(
                    tokens=tokens,
            		segment_ids=segment_ids,
            		is_random_next=is_random_next,
            		masked_lm_positions=masked_lm_positions,
            		masked_lm_labels=masked_lm_labels)
        		instances.append(instance)
      		current_chunk = []
      		current_length = 0
    	i += 1
 
	return instances
```

**打mask**

```python
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # wwm
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
      		cand_indexes[-1].append(i)
    	else:
            cand_indexes.append([i])

  	rng.shuffle(cand_indexes)
    output_tokens = list(tokens)
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

  	masked_lms = []
    # mask过的位置
  	covered_indexes = set()
  	for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # wwm
    	if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        # 判断是否重复mask，continue
    	is_any_index_covered = False
    	for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
    	if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

      	masked_token = None
      	# 80%用[MASK]替换
      	if rng.random() < 0.8:
            masked_token = "[MASK]"
      	else:
        	# 10%不替换
        	if rng.random() < 0.5:
         		 masked_token = tokens[index]
        	# 10%随机替换
        	else:
          		masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
      	output_tokens[index] = masked_token
        # 添加样本
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  	masked_lms = sorted(masked_lms, key=lambda x: x.index)
    
    # mask的位置、label
  	masked_lm_positions = []
  	masked_lm_labels = []
  	for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    
    return (output_tokens, masked_lm_positions, masked_lm_labels)
```



## 预训练

**主函数**

```python
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    # 获取输入文件
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))
    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)
        
    # 配置tpu
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver, master=FLAGS.master,
        model_dir=FLAGS.output_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    
    # 创建bert，config、checkpoint、lr、steps
    model_fn = model_fn_builder(
        bert_config=bert_config, init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate, num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps, use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    # 定义训练器
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu, model_fn=model_fn, config=run_config,
        train_batch_size=FLAGS.train_batch_size, eval_batch_size=FLAGS.eval_batch_size)
    
    # 训练
    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        # 读取tf输入
        train_input_fn = input_fn_builder(
            input_files=input_files, max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq, is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
    # 评估
    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
        eval_input_fn = input_fn_builder(
            input_files=input_files, max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq, is_training=False)
        result = estimator.evaluate(input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
```

**创建bert**

```python
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        # 特征信息，输入（ids、mask、句对ids）、mask信息（位置、labels、权重）、nsp信息
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # bert model
        model = modeling.BertModel(
            config=bert_config, is_training=is_training,
            input_ids=input_ids, input_mask=input_mask,
            token_type_ids=segment_ids, use_one_hot_embeddings=use_one_hot_embeddings
        # 损失函数
        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config, model.get_sequence_output(), model.get_embedding_table(),
            masked_lm_positions, masked_lm_ids, masked_lm_weights)
        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
            bert_config, model.get_pooled_output(), next_sentence_labels)
        total_loss = masked_lm_loss + next_sentence_loss
            
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        # 载入模型
        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        # 训练
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate,
                num_train_steps, num_warmup_steps, use_tpu)
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss,
                train_op=train_op, scaffold_fn=scaffold_fn)
        # 评估
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                          masked_lm_weights, next_sentence_example_loss,
                          next_sentence_log_probs, next_sentence_labels):
                # mlm logits
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs,[-1, masked_lm_log_probs.shape[-1]])
                # 预测值
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
                # labels
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                # 计算准确率
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                # 损失加权平均
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)
                # nsp
                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
                next_sentence_predictions = tf.argmax(
                    next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.metrics.accuracy(
                    labels=next_sentence_labels, predictions=next_sentence_predictions)
                next_sentence_mean_loss = tf.metrics.mean(
                    values=next_sentence_example_loss)
                # 返回准确率+损失
                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                }
            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels
            ])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn
```

**读取tf输入**

```python
def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    def input_fn(params):
        batch_size = params["batch_size"]
        name_to_features = {
            "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions": tf.FixedLenFeature(
                [max_predictions_per_seq], tf.int64),
            "masked_lm_ids": tf.FixedLenFeature(
                [max_predictions_per_seq], tf.int64),
            "masked_lm_weights": tf.FixedLenFeature(
                [max_predictions_per_seq], tf.float32),
            "next_sentence_labels": tf.FixedLenFeature([1], tf.int64),
        }
        
        if is_training:
            # 读取输入
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))
            # 并行读取
            cycle_length = min(num_cpu_threads, len(input_files))
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn
```

