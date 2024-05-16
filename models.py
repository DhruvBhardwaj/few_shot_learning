import torch
from torch import nn
import torch.nn.functional as F

class convNet(nn.Module):
    def __init__(self, cfg, device):
        super(convNet,self).__init__()

        self.config = cfg
        self.device = device
        layers =[]

        channel_set=self.config['channels']
        for i in range(self.config['num_conv_blocks']):
            if i==0:
                layers.append(nn.Conv2d(self.config['in_channels'],channel_set[i][1],3, bias=False))
            else:
                layers.append(nn.Conv2d(channel_set[i][0],channel_set[i][1],3, bias=False))
            
            layers.append(nn.BatchNorm2d(channel_set[i][1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            layers.append(nn.MaxPool2d(2))

        self.net = nn.ModuleList(layers)
        print('-'*59)
        print('ConvNet')
        print(self.net)
        print('-'*59)

    def forward(self,x):

        for _, l in enumerate(self.net):
            x = l(x)            

        return x

class baseClassifier(nn.Module):
    def __init__(self, cfg, device):
        super(baseClassifier,self).__init__()
        self.config = cfg
        self.device = device

        print('BASE_CLASSIFIER')
        self.convnet = convNet(cfg,device)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(cfg['base_in_features'],cfg['out_classes'])
        
        print('Linear')
        print(self.flat)
        print(self.dense)
        print('-'*59)

    def forward(self,x):  
        x = self.flat(self.convnet(x))
        return self.dense(x)

    def get_feature(self,x):
        return self.flat(self.convnet(x))

class binaryClassifier(nn.Module):
    def __init__(self, cfg, device,pre_trained_file=None):
        super(binaryClassifier,self).__init__()
        self.config = cfg
        self.device = device

        print('BINARY_CLASSIFIER')
        self.convnet = convNet(cfg,device)
        if(pre_trained_file is not None):
            print('loading',pre_trained_file)
            self.load_wts(pre_trained_file)
            for param in self.convnet.parameters():
                param.requires_grad = False

        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(cfg['base_in_features'],10)
        self.dense2 = nn.Linear(10,2)
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print ('--',name, param.data.sum())

        print('Linear')
        print(self.flat)
        print(self.dense1)
        print(self.dense2)
        print('-'*59)        

    def forward(self,x):  
        x = self.flat(self.convnet(x))
        return self.dense2(self.dense1(x))
    
    def load_wts(self,chkpt_file):
        pretrained_dict = torch.load(chkpt_file)['model_state_dict']        
        model_dict = self.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}        

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.load_state_dict(model_dict)

        return

class pentaClassifier(nn.Module):
    def __init__(self, cfg, device,pre_trained_file=None):
        super(pentaClassifier,self).__init__()
        self.config = cfg
        self.device = device

        print('BINARY_CLASSIFIER')
        self.convnet = convNet(cfg,device)
        if(pre_trained_file is not None):
            print('loading',pre_trained_file)
            self.load_wts(pre_trained_file)
            for param in self.convnet.parameters():
                param.requires_grad = False

        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(cfg['base_in_features'],10)
        self.dense2 = nn.Linear(10,5)
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print ('--',name, param.data.sum())

        print('Linear')
        print(self.flat)
        print(self.dense1)
        print(self.dense2)
        print('-'*59)

    def forward(self,x):  
        x = self.flat(self.convnet(x))
        return self.dense2(self.dense1(x))

    def load_wts(self,chkpt_file):
        pretrained_dict = torch.load(chkpt_file)['model_state_dict']        
        model_dict = self.state_dict()
        
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}        

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.load_state_dict(model_dict)

        return

class metaClassifier(nn.Module):
    def __init__(self, cfg, device):
        super(metaClassifier,self).__init__()

        self.config = cfg
        self.device = device        
        layers =[]        

        channel_set=self.config['channels']
        for i in range(self.config['num_conv_blocks']):
            if i==0:
                layers.append(nn.Conv2d(self.config['in_channels'],channel_set[i][1],kernel_size=3,padding=1, bias=False))
            else:
                layers.append(nn.Conv2d(channel_set[i][0],channel_set[i][1],kernel_size=3,padding=1, bias=False))
            
            layers.append(nn.BatchNorm2d(channel_set[i][1]))
            layers.append(nn.ReLU())
            if(i<self.config['num_conv_blocks']-1):            
                layers.append(nn.MaxPool2d(2))

        self.net = nn.ModuleList(layers)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(cfg['base_in_features'],cfg['out_classes'])
        print('-'*59)
        print('metaClassifier')
        print(self.net)
        print(self.flat)
        print(self.dense)
        print('-'*59)        

    def forward(self,x):

        for _, l in enumerate(self.net):
            x = l(x)            
        x = self.dense(self.flat(x))
        return x    

    def print_params(self,query):
        for name, param in self.named_parameters():
            if param.requires_grad and (query in name):
                print ('--',name, param.data.sum())
        return

    def print_grad_norm(self):
        total_norm=0.
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** (1. / 2)
        print('-',total_norm)
        return

class protoClassifier(nn.Module):
    def __init__(self, cfg, device):
        super(protoClassifier,self).__init__()

        self.config = cfg
        self.device = device        
        layers =[]        

        channel_set=self.config['channels']
        for i in range(self.config['num_conv_blocks']):
            if i==0:
                layers.append(nn.Conv2d(self.config['in_channels'],channel_set[i][1],kernel_size=3,padding=1, bias=False))
            else:
                layers.append(nn.Conv2d(channel_set[i][0],channel_set[i][1],kernel_size=3,padding=1, bias=False))
            
            layers.append(nn.BatchNorm2d(channel_set[i][1]))
            layers.append(nn.ReLU())
            if(i<self.config['num_conv_blocks']-1):            
                layers.append(nn.MaxPool2d(2))

        self.net = nn.ModuleList(layers)
        self.flat = nn.Flatten()
        
        print('-'*59)
        print('protoClassifier')
        print(self.net)
        print(self.flat)
        
        print('-'*59)        
        

    def compute_prototype(self,x, y):
        for _, l in enumerate(self.net):
            x = l(x)            
        x = self.flat(x)
        classes,_ = y.unique().sort()

        prot=[]
        for c in classes:            
            prot.append(x[torch.where(y==c)[0],:].mean(dim=0).unsqueeze(0))

        self.prototypes = torch.cat(prot,dim=0)        
        return

    def forward(self,x):

        for _, l in enumerate(self.net):
            x = l(x)            
        x = self.flat(x)
        
        # x is B x EMBED_DIM
        # xref is NUM_EMBED X EMBED_DIM

        xref = self.prototypes
        x_norm2 = torch.linalg.norm(x, dim=1,keepdim=True)**2 #[B x 1]
        x_norm2 = x_norm2.expand(-1,xref.size(0)) #[BC x NUM_EMBED]

        xref_norm2 = torch.linalg.norm(xref, dim=1,keepdim=True)**2 #[NUM_EMBED x 1]
        xref_norm2 = xref_norm2.expand(-1, x.size(0)) #[NUM_EMBED x B]
        xref_norm2 = torch.transpose(xref_norm2, 0, 1) #[B X NUM_EMBED]        
        
        dist = x_norm2 + xref_norm2 - 2*torch.matmul(x, torch.transpose(xref, 0 , 1))                 
        return -1*dist

if __name__ == '__main__':    
    from config import cfg
    #m = convNet(cfg['model'],'cpu')
    # m = baseClassifier(cfg['model'],'cpu')
    # x = torch.randn(64,3,32,32)
    # y = m(x)
    # print(y.size())
    m = binaryClassifier(cfg['model'],'cuda:3','chkpt/base_classifier_expt_1a_1.chk.pt')
    x = torch.randn(64,3,32,32)
    print(m(x).size())

    # m = metaClassifier(cfg['model'],'cpu')
    # x = torch.randn(25,1,28,28)
    # y = m(x)
    # print(y.size())

    # m = protoClassifier(cfg['model'],'cpu')
    # x = torch.randn(6,1,28,28)
    # y = torch.tensor([1,0,2,3,3,4])
    # m.compute_prototype(x,y)
    # y_hat = m(x)# + 0.001*torch.randn(6,1,28,28))
    # print(y_hat)
    # print(y_hat.size())
