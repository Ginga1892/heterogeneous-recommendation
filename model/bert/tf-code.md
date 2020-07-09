# BERT TensorFlow源码

## 预处理

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

**主函数**

```python
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # 定义分词
    tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    
    # 获取输入
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

