import torch 
import torch.nn as nn

class FCN(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(FCN, self).__init__()
        
        # Block 단위로 만듬 2D Conv + BatchNorm + LeakyReLU로 정의
        def block(in_channels, out_channels, kernel_size = 3, stride =1 , padding = 1, bias = True):
            layers = []
            ## Conv 2D
            layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)]
            ## Batch Norm
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            ## LeakyReLU
            layers += [nn.LeakyReLU()]
            
            cbr = nn.Sequential(*layers)
            return cbr     
        
        # 1->4->16->64 --> 1 
        self.enc1 = block(in_channels= 1, out_channels=4)
        self.enc2 = block(in_channels= 4, out_channels= 16)
        self.enc3 = block(in_channels= 16, out_channels= 64)
        self.fc = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, stride = 1, padding = 0, bias = True)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        out = self.fc(enc3)
        return out
class UNet_2d(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(UNet_2d, self).__init__() 
        
        # 네트워크에서 반복적으로 사용하는 Convolution + BatchNormalize + Relu 를 하나의 block으로 정의
        def CBR2d(in_channels, out_channels, kernel_size = 3, stride =1, padding = 1, bias = True):
            layers = []
            ## conv2d
            layers += [nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_szie = kernel_size, stride = stride, padding = padding, bias = bias)]
            ## batchnorm2d
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            ## ReLU
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers)
            
            return cbr
        
        ## Encoder 
        self.enc1_1 = CBR2d(in_channels= 1, out_channels= 64)
        self.enc1_2 = CBR2d(in_channels= 64, out_channels= 64)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc2_1 = CBR2d(in_channels= 64, out_channels= 128)
        self.enc2_2 = CBR2d(in_channels= 128, out_channels= 128)
        
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc3_1 = CBR2d(in_channels= 128, out_channels= 256)
        self.enc3_2 = CBR2d(in_channels= 256, out_channels= 256)
        
        self.pool3 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc4_1 = CBR2d(in_channels= 256, out_channels= 512)
        self.enc4_2 = CBR2d(in_channels= 512, out_channels= 512)
        
        self.pool4 = nn.MaxPool2d(kernel_size = 2)
        
        self.enc5_1 = CBR2d(in_channels = 512, out_channels = 1024)
        ## Decoder 
        self.dec5_1 = CBR2d(in_channels= 1024, out_channels = 512)
        
        self.unpool4 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec4_2 = CBR2d(in_channels= 2 * 512, out_channels= 512)
        self.dec4_1 = CBR2d(in_channels= 512, out_channels= 256)
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec3_2 = CBR2d(in_channels= 2 * 256, out_channels= 256)
        self.dec3_1 = CBR2d(in_channels= 256, out_channels= 128)
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec2_2 = CBR2d(in_channels= 2 * 128, out_channels= 128)
        self.dec2_1 = CBR2d(in_channels= 128, out_channels= 64)
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size = 2)
        
        self.dec1_2 = CBR2d(in_channels= 2 * 64, out_channels= 64)
        self.dec1_1 = CBR2d(in_channels= 64, out_channels= 64)
        
        self.fc = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size =1, stride = 1, padding = 0, bias = True)
        
    def forward(self, x):
        # Channel : 1 --> 64
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        # Channel : 64 --> 128
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        # Channel : 128 --> 256
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        # Channel : 256 --> 512
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        # Channel : 512 --> 1024 --> 512
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        
        # Channel : 1024 --> 512 
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim = 1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        # Channel : 512 --> 256
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim = 1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        # Channel : 256 --> 128
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        # Channel : 128 --> 64
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        # Channel -> FCL 로 전환 
        out = self.fc(dec1_1) 
        
        return out 
class UNet_3d(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(UNet_3d, self).__init__() 
        # 네트워크에서 반복적으로 사용하는 Convolution + BatchNormalize + Relu 를 하나의 block으로 정의
        def CBR3d(in_channels, out_channels, kernel_size = 3, stride =1, padding = 1, bias = True):
            layers = []
            ## conv3d
            layers += [nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_szie = kernel_size, stride = stride, padding = padding, bias = bias)]
            ## batchnorm2d
            layers += [nn.BatchNorm3d(num_features = out_channels)]
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers)
            
            return cbr
        
        ## Encoder 
        self.enc1_1 = CBR3d(in_channels= 1, out_channels= 32)
        self.enc1_2 = CBR3d(in_channels= 32, out_channels= 32)
        
        self.pool1 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc2_1 = CBR3d(in_channels= 32, out_channels= 64)
        self.enc2_2 = CBR3d(in_channels= 64, out_channels= 64)
        
        self.pool2 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc3_1 = CBR3d(in_channels= 64, out_channels= 128)
        self.enc3_2 = CBR3d(in_channels= 128, out_channels= 128)
        
        self.pool3 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc4_1 = CBR3d(in_channels= 128, out_channels= 256)
        self.enc4_2 = CBR3d(in_channels= 256, out_channels= 256)
        
        self.pool4 = nn.MaxPool3d(kernel_size = 2)
        
        self.enc5_1 = CBR3d(in_channels = 256, out_channels = 512)
        ## Decoder 
        self.dec5_1 = CBR3d(in_channels= 512, out_channels = 256)
        
        self.unpool4 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec4_2 = CBR3d(in_channels= 2 * 256, out_channels= 256)
        self.dec4_1 = CBR3d(in_channels= 256, out_channels= 128)
        
        self.unpool3 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec3_2 = CBR3d(in_channels= 2 * 128, out_channels= 128)
        self.dec3_1 = CBR3d(in_channels= 128, out_channels= 64)
        
        self.unpool2 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec2_2 = CBR3d(in_channels= 2 * 64, out_channels= 64)
        self.dec2_1 = CBR3d(in_channels= 64, out_channels= 32)
        
        self.unpool1 = nn.MaxUnpool3d(kernel_size = 2)
        
        self.dec1_2 = CBR3d(in_channels= 2 * 32, out_channels= 32)
        self.dec1_1 = CBR3d(in_channels= 32, out_channels= 32)
        
        self.fc = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size =1, stride = 1, padding = 0, bias = True)
        
    def forward(self, x):
        # Channel : 1 --> 32
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)
        
        # Channel : 32 --> 64
        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)
        
        # Channel : 64 --> 128
        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)
        
        # Channel : 128 --> 256
        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)
        
        # Channel : 256 --> 512 --> 256
        enc5_1 = self.enc5_1(pool4)
        dec5_1 = self.dec5_1(enc5_1)
        
        # Channel : 512 --> 256 
        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim = 1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)
        
        # Channel : 256 --> 128
        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim = 1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)
        
        # Channel : 128 --> 64
        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim = 1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)
        
        # Channel : 64 --> 32
        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim = 1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        # Channel -> FCL 로 전환 
        out = self.fc(dec1_1) 
        
        return out
class Conv_AutoEncoder_3D(nn.Module):
    # input = channels * width * height  
    def __init__(self):
        super(Conv_AutoEncoder_3D, self).__init__()
        ## encoder 
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels= 1, out_channels = 16, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 16),
            nn.ReLU(),
            nn.Conv3d(in_channels= 16, out_channels = 32, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 32), #BatchnNorm3d 에서 32로 가야됨 
            nn.ReLU(),
            nn.Conv3d(in_channels= 32, out_channels = 64, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 64),
            nn.ReLU(),
        )
        ## decoder 
        self.decoder = nn.Sequential(
            nn.Conv3d(in_channels= 64, out_channels = 32, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 32),
            nn.ReLU(),
            nn.Conv3d(in_channels= 32, out_channels = 16, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 16), #BatchnNorm3d 에서 32로 가야됨 
            nn.ReLU(),
            nn.Conv3d(in_channels= 16, out_channels = 1, kernel_size = 5, bias = True),
            nn.BatchNorm3d(num_features = 1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1 , hidden_dim2):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),            
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = x.view(x.size(0), -1) #flatten 시키기
        out = self.encoder(out)
        out = self.decoder(out)
        out = out.view(x.size()) #input이미지와 사이즈 동일하게 다시 만들기
        return out 
    
    ## hidden state 값 = latent vector 확인 
    def hidden_state(self, x): 
        return self.encoder(x)