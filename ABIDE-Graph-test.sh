python main.py --dataname ABIDE --marl_mode 2  --n_head 3 --n_hidden 22 --MP_mode GCN --mode local-global --GC_mode weighted-cosine \
          --reg 0.05 --lr 0.002 --dropout 0.357 \
          --nlayer 2 --gl_attn_layer 2 --gl_lin_layer 0 \
          --CL_weight 0.65 --CL_rate 0.55 --CL_mse 0.6 \
          --CL_mode random_mask_edge


