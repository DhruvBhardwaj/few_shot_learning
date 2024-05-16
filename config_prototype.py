cfg={        
        'training':{
            'batch_size':32,
            'num_episodes':200,
            'num_test_episodes':1000,
            'alpha':0.1,            
            'N-way':5,
            'K-shot':5,
            'save_path':'./logs/',
            'data_path':'./data/',            
            'chkpt_path':'./chkpt/',            
            'chkpt_file':'expt_2a.chk.pt',            
            'load_from_chkpt':False
        },
        'model':{
            'in_channels':1,
            'out_classes':5,
            'num_conv_blocks':5,
            'channels':[
                [1,2],
                [2,4],
                [4,8],
                [8,16],
                [16,32]
            ],
            'base_in_features':32
        }
    }