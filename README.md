allennlp实现AcademicPaperClassifier
==================================

## 参考

https://zhuanlan.zhihu.com/p/73469009

https://github.com/allenai/allennlp-as-a-library-example


## 在此基础上了做了修改，添加自定义特征的embedding向量

由于allennlp默认的embedding方式有token-level、character-level、elmo、bert等等，但是不支持自定义的embedding。

论文中有时候会提取一些其他特征，例如 https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs 除了提取了word-level 和 character-level的embedding特征，还提取了word-case相关的特征 : allCaps，upperInitial，lowercase，mixedCaps，noinfo。

**解决方法**

word和character都会建立一个词表，对word或者char进行编号，然后进行embedding操作，对于word-case的特征，可以将其看成“word”，与word同时进行编号，但是采用不同的text_field_embedder方式进行操作，从而实现自定义embedding。具体方式如下：

- 配置文件

原来的json文件：

```json
{
  "dataset_reader": {
    "type": "s2_papers"
  },
  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/train.jsonl",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/dev.jsonl",
  "model": {
    "type": "paper_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": false
        }
      }
    },
    "title_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 100,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 400,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["abstract", "num_tokens"], ["title", "num_tokens"]],
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}
```

修改后的json文件：

```json
{
  "dataset_reader": {
    "type": "s2_papers",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters"
      }
    }
  },
  "train_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/train.jsonl",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/academic-papers-example/dev.jsonl",
  "model": {
    "type": "paper_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": false
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 128,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu"
            }
        }
      }
    },
    "text_field_embedder_other": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 30
        }
      }
    },
    "title_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 228,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "abstract_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 228,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },
    "other_encoder": {
      "type": "lstm",
      "bidirectional": true,
      "input_size": 30,
      "hidden_size": 20,
      "num_layers": 1,
      "dropout": 0.2
    },
    "classifier_feedforward": {
      "input_dim": 440,
      "num_layers": 2,
      "hidden_dims": [200, 3],
      "activations": ["relu", "linear"],
      "dropout": [0.2, 0.0]
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["abstract", "num_tokens"], ["title", "num_tokens"]],
    "batch_size": 64
  },
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": -1,
    "grad_clipping": 5.0,
    "validation_metric": "+accuracy",
    "optimizer": {
      "type": "adagrad"
    }
  }
}

```

修改后的json文件定义了两个text_field_embedder，一个是针对title和abstract的word 和 character embedding；一个是针对word-case 特征的embedding


- 特征提取和数据流

semantic_scholar_papers.py文件里增加了word-case特征提取，这里只增加了word是否首字母是大写的特征，如下所示：

```
other = [Token("all_title") if token.istitle() else Token("no_all_title") for token in title]


other_field = TextField(other, self._token_indexers)
fields = {'title': title_field, 'abstract': abstract_field, 'other': other_field}
```

- model

academic_paper_classifier.py文件里关于word-case特征的embedding方式如下：

```python
other_tokens = {}
other_tokens["tokens"] = other["tokens"]
embedded_other = self.text_field_embedder_other(other_tokens)
other_mask = util.get_text_field_mask(other_tokens)
encoded_other = self.other_encoder(embedded_other, other_mask)
```

由于 token_indexers里有token_characters，所以传过来的other是一个字典，包括 tokens 和 token_characters，我们只需要tokens，所以只提取tokens放入text_field_embedder_other做embedding，然后传入对应的model，进行下一步操作，在输出文件的vocabulary里，可以看到tokens.txt里包含word-case的两个特征：all_title 和 no_all_title
