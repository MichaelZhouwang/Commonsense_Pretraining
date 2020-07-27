```

```

```
bash csqa_eval.sh 1 original_csqa -1 16 1 > ori.1.result &
watch tail ori.1.result

bash csqa_eval.sh 0 original_csqa -1 16 42 > ori.42.result &
watch tail ori.42.result

bash csqa_eval.sh 1 original_csqa -1 16 1337 > ori.1337.result &
watch tail ori.1337.result

bash csqa_eval.sh 0 bart_packed -1 16 1 > bart.1.result &
watch tail bart.1.result
bash csqa_eval.sh 2 bart_packed -1 16 42 > bart.42.result &
watch tail bart.42.result
  
```


```
bash csqa_eval.sh 1 original_csqa -1 16 > ori.result &
watch tail ori.result

bash csqa_eval.sh 2 bart_packed -1 16 > bart.result &
watch tail bart.result
 
```