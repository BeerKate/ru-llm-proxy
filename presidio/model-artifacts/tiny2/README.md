# Файлы модели для профиля `tiny2`

Модель rubert-tiny2 для маскировки не опубликована на Hugging Face. Чтобы собрать
образ Analyzer с ней, положите сюда шесть файлов из манифеста
`presidio/model_manifest.tiny2.json`:

```
config.json
model.safetensors
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.txt
```

Получите файлы checkpoint `tiny2-mix-a-s1-ft-spacy-free-20261013-a-e2-b2` у владельца артефакта и
положите их в эту папку. Расположение исходного архива и постоянное хранилище
весов должны быть согласованы до промышленной сборки.

Затем соберите и запустите стек с профилем
([развёртывание](../../../docs/deployment.md#tiny2)):

```bash
make build ANALYZER_PROFILE=tiny2
make up STACK=litellm-presidio ANALYZER_PROFILE=tiny2
```

При сборке каждый файл сверяется с размером и SHA-256 из манифеста; при
несовпадении сборка останавливается. При запуске сервис ещё раз проверяет файлы
и сверяет модель с профилем. Веса в git не добавляются (см. `.gitignore`).
