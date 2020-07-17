epochs=20
lr=0.005 #0.0008
hidden=200
path_len=3
dropout=0.4
cudaid=3
weight_decay=5e-4
dataset='science_wordnet'
fp="../data_${dataset}_en_0.2"
encode_dep='rnn'
encode_prop='linear'
with_attn=1
minus_only=1
layers=1
taxi_feature=1
lambda1=0.1
lambda2=0.15
load_gcn=1
load_score=1
model_name=fuse_${dataset}_0.2_taxi_${lambda1}_${lambda2}_${path_len}
python train_fuse.py --epochs $epochs \
                   --lr $lr \
                   --cudaid $cudaid \
                   --dropout $dropout \
                   --hidden $hidden \
                   --weight_decay $weight_decay \
                   --fp $fp \
	     --path_len $path_len \
                   --encode_dep $encode_dep \
                   --encode_prop $encode_prop \
                   --minus_only $minus_only \
                   --layers $layers \
                   --taxi_feature $taxi_feature \
	     --model_name $model_name \
 	     --with_attn $with_attn \
	      --lambda1 $lambda1 \
	      --lambda2 $lambda2 \
	      --load_gcn $load_gcn \
	      --load_score $load_score \

path_len=3
epochs=20
lambda1=0.15
lambda2=0.1
load_gcn=1
load_score=1
model_name=fuse_${dataset}_0.2_taxi_${lambda1}_${lambda2}_${path_len}
python train_fuse.py --epochs $epochs \
                   --lr $lr \
                   --cudaid $cudaid \
                   --dropout $dropout \
                   --hidden $hidden \
                   --weight_decay $weight_decay \
                   --fp $fp \
	     --path_len $path_len \
                   --encode_dep $encode_dep \
                   --encode_prop $encode_prop \
                   --minus_only $minus_only \
                   --layers $layers \
                   --taxi_feature $taxi_feature \
	     --model_name $model_name \
 	     --with_attn $with_attn \
	      --lambda1 $lambda1 \
	      --lambda2 $lambda2 \
	      --load_gcn $load_gcn\
	      --load_score $load_score\