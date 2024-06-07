


conda activate vit2 

############################ CIFAR-100 ############################
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk 
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q 
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 #baseline 

python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.3 #baseline + attention dropout 
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropout 0.3 # both attention and qk dropout
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropout 0.3 # both attention and k dropout
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --dropout 0.3 # both attention and q dropout

python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.3 #baseline + dropkey
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.3 --dropkey --mask_ratio 0.3 #baseline + dropkey + attntion dropout
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropkey --mask_ratio 0.3 #qk + dropkey 
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropkey --mask_ratio 0.3 #k + dropkey 
# running: dropkey
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.1 #baseline + dropkey
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.2 #baseline + dropkey
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.4 #baseline + dropkey
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.5 #baseline + dropkey

python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.1 #baseline + attention dropout # repaet
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.2 #baseline + attention dropout 
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.4 #baseline + attention dropout 
python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropout 0.5 #baseline + attention dropout 

# running on repeat
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.1 #baseline + attention dropout # repaet
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.2 #baseline + attention dropout # repeat
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.3 #baseline + attention dropout # repeat
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.4 #baseline + attention dropout # repeat
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.5 #baseline + attention dropout # repeat
# 


# new runnings 15th Nov: 400 epochs
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None # baseline


CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on qk 

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on k 

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on q 


#######
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --dropkey --mask_ratio 0.3 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropkey --mask_ratio 0.3 #baseline + qk +  dropkey

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.1 --mlp_dropout 0.1 #baseline + attention dropout # repaet
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.2 --mlp_dropout 0.2 #baseline + attention dropout 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.3 --mlp_dropout 0.3 #baseline + attention dropout 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.4 --mlp_dropout 0.4 #baseline + attention dropout 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.5 --mlp_dropout 0.5 #baseline + attention dropout 


### running: Monday 13th 

CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --mlp_dropout 0.1 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --mlp_dropout 0.1 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --mlp_dropout 0.1 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --mlp_dropout 0.2 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --mlp_dropout 0.2 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --mlp_dropout 0.2 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --mlp_dropout 0.3 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --mlp_dropout 0.3 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --mlp_dropout 0.3 #baseline + mlp dropout + qk

CUDA_VISIBLE_DEVICES=6 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --mlp_dropout 0.1 --attn_dropout 0.1 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --mlp_dropout 0.1 --attn_dropout 0.1 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --mlp_dropout 0.1 --attn_dropout 0.1 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --mlp_dropout 0.2 --attn_dropout 0.2 #baseline + mlp dropout + q

CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --mlp_dropout 0.2 --attn_dropout 0.2 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --mlp_dropout 0.2 --attn_dropout 0.2 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --mlp_dropout 0.3 --attn_dropout 0.3 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --mlp_dropout 0.3 --attn_dropout 0.3 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --mlp_dropout 0.3 --attn_dropout 0.3 #baseline + mlp dropout + qk

CUDA_VISIBLE_DEVICES=4 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --dropkey --mask_ratio 0.1 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --dropkey --mask_ratio 0.1 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --dropkey --mask_ratio 0.2 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --dropkey --mask_ratio 0.2 #baseline + k +  dropkey

CUDA_VISIBLE_DEVICES=7 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment  --dropout_on None --dropkey --mask_ratio 0.1 --attn_dropout 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment  --dropout_on None --dropkey --mask_ratio 0.2 --attn_dropout 0.2
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment  --dropout_on None --dropkey --mask_ratio 0.3 --attn_dropout 0.3

CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q 
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q 
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q 
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k 
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k 
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k 
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk 
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk 
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk

# Thursday 16th
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --attn_dropout 0.1 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --attn_dropout 0.1 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --attn_dropout 0.1 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --attn_dropout 0.2 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --attn_dropout 0.2 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --attn_dropout 0.2 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --attn_dropout 0.3 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --attn_dropout 0.3 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --attn_dropout 0.3 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on q --attn_dropout 0.4 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on k --attn_dropout 0.4 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on qk --attn_dropout 0.4 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on q --attn_dropout 0.5 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on k --attn_dropout 0.5 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on qk --attn_dropout 0.5 #baseline + attention dropout + qk

CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.1 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.2 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.3 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.4 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.5 #baseline + attention dropout + k


#### Friday: 17th 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropkey --mask_ratio 0.3 #baseline + k +  dropkey

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.4 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.5 #baseline + k +  dropkey


############################ CIFAR-10 ############################
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk 
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 #baseline 
# running:
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.3 #baseline + attention dropout 
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropout 0.3 # both attention and qk dropout
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropout 0.3 # both attention and qk dropout
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --dropout 0.3 # both attention and qk dropout
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on qk 
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk 
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.3 #baseline + dropkey
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.3 --dropkey --mask_ratio 0.3 #baseline + dropkey + attntion dropout
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropkey --mask_ratio 0.3 #qk + dropkey 
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropkey --mask_ratio 0.3 #k + dropkey 
# running: dropkey
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.1 #baseline + dropkey
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.2 #baseline + dropkey
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.4 #baseline + dropkey
python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropout 0.0 --dropkey --mask_ratio 0.5 #baseline + dropkey


# new runnings 15th Nov:
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None # baseline

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on qk 

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on k 

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on q 


############

CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --dropkey --mask_ratio 0.3 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --attn_dropout 0.3 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --attn_dropout 0.3 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --attn_dropout 0.3 #baseline + attention dropout + qk

CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.1 --mlp_dropout 0.1 #baseline + attention dropout # repaet
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.2 --mlp_dropout 0.2 #baseline + attention dropout 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.3 --mlp_dropout 0.3 #baseline + attention dropout 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.4 --mlp_dropout 0.4 #baseline + attention dropout 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.5 --mlp_dropout 0.5 #baseline + attention dropout 

conda activate vit2 
#########  running: Monday 13th CIFAR 10
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --attn_dropout 0.1 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --attn_dropout 0.1 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --attn_dropout 0.1 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --attn_dropout 0.2 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --attn_dropout 0.2 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --attn_dropout 0.2 #baseline + attention dropout + qk

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --mlp_dropout 0.1 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --mlp_dropout 0.1 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --mlp_dropout 0.1 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --mlp_dropout 0.2 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --mlp_dropout 0.2 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --mlp_dropout 0.2 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --mlp_dropout 0.3 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --mlp_dropout 0.3 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --mlp_dropout 0.3 #baseline + mlp dropout + qk

CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --mlp_dropout 0.1 --attn_dropout 0.1 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --mlp_dropout 0.1 --attn_dropout 0.1 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --mlp_dropout 0.1 --attn_dropout 0.1 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --mlp_dropout 0.2 --attn_dropout 0.2 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --mlp_dropout 0.2 --attn_dropout 0.2 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --mlp_dropout 0.2 --attn_dropout 0.2 #baseline + mlp dropout + qk
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --mlp_dropout 0.3 --attn_dropout 0.3 #baseline + mlp dropout + q
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --mlp_dropout 0.3 --attn_dropout 0.3 #baseline + mlp dropout + k
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --mlp_dropout 0.3 --attn_dropout 0.3 #baseline + mlp dropout + qk

CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --dropkey --mask_ratio 0.1 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --dropkey --mask_ratio 0.1 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --dropkey --mask_ratio 0.2 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --dropkey --mask_ratio 0.2 #baseline + k +  dropkey

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment  --dropout_on None --dropkey --mask_ratio 0.1 --attn_dropout 0.1
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment  --dropout_on None --dropkey --mask_ratio 0.2 --attn_dropout 0.2
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment  --dropout_on None --dropkey --mask_ratio 0.3 --attn_dropout 0.3

CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q # ran once before with the new dropout (on second repeat)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k # ran once before with the new dropout (on second repeat)
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk # ran once before with the new dropout (on second repeat)

# Thursday 16th
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --attn_dropout 0.1 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --attn_dropout 0.1 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --attn_dropout 0.1 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --attn_dropout 0.2 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --attn_dropout 0.2 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --attn_dropout 0.2 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --attn_dropout 0.3 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --attn_dropout 0.3 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --attn_dropout 0.3 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=6 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on q --attn_dropout 0.4 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on k --attn_dropout 0.4 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=7 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.4 --dropout_on qk --attn_dropout 0.4 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on q --attn_dropout 0.5 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on k --attn_dropout 0.5 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.5 --dropout_on qk --attn_dropout 0.5 #baseline + attention dropout + qk

CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.1 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.2 #baseline + attention dropout + k
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.3 #baseline + attention dropout + qk
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.4 #baseline + attention dropout + q
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --attn_dropout 0.5 #baseline + attention dropout + k

#### Friday: 17th 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk --dropkey --mask_ratio 0.3 #baseline + k +  dropkey

CUDA_VISIBLE_DEVICES=1 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.1 #baseline + q +  dropkey
# next: running
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.2 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.3 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.4 #baseline + k +  dropkey
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment --dropout_on None --dropkey --mask_ratio 0.5 #baseline + k +  dropkey


# 3 Dec running
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on q 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.1 --dropout_on qk 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on q 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on k 
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.2 --dropout_on qk 
CUDA_VISIBLE_DEVICES=1 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on q # ran once before with the new dropout (on second repeat)
CUDA_VISIBLE_DEVICES=2 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on k # ran once before with the new dropout (on second repeat)
CUDA_VISIBLE_DEVICES=3 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment --qk_dropout 0.3 --dropout_on qk # ran once before with the new dropout (on second repeat)

# running:
CUDA_VISIBLE_DEVICES=4 python main.py --dataset c100 --num-classes 100 --label-smoothing --autoaugment  # baseline 
CUDA_VISIBLE_DEVICES=5 python main.py --dataset c10 --num-classes 10 --label-smoothing --autoaugment  # baseline 


