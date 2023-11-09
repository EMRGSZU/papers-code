from math import inf
import torch
import torch.nn as nn
from collections import deque
from tools import common
import torch.nn.functional as F

class ADMM_RED_UNFOLD(nn.Module):

    def __init__(self, ratio, n_iteration = 9, n_channel = 32, block_size = 32, sample_matrix = None, init_matrix = None):
        super(ADMM_RED_UNFOLD, self).__init__()
        self.block_size = block_size
        self.ratio = ratio
        self.iteration = n_iteration
        self.channel = n_channel

        #生成采样矩阵
        matrix, matrix_t = self.generate_matrix(self.block_size, ratio)

        self.sample = nn.Parameter(matrix)
        self.init = nn.Parameter(matrix_t)
        
        if torch.is_tensor(sample_matrix) and torch.is_tensor(init_matrix):
            self.sample = nn.Parameter(sample_matrix)
            self.init = nn.Parameter(init_matrix)

        self.unfold = nn.Unfold(32, stride=32)
        self.fold = nn.Fold((96, 96), 32, stride=32)

        #x_infs和u_infs分别记录ICS在x_block和u_block迭代时队列内信息的数量
        x_infs = [4, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9]
        u_infs = [3, 7, 11, 12, 12, 12, 12, 12, 12, 12, 12]
        self.xblocks = nn.ModuleList([XBlock(self.channel, x_infs[i]) for i in range(self.iteration)])
        self.ublocks = nn.ModuleList([UBlock(self.channel, u_infs[i]) for i in range(self.iteration)])
        self.deblocker = DeBlocker(self.channel)
        self.ics = ICS()

    def generate_matrix(self, block_size, ratio):
        n_cols = block_size ** 2
        n_rows = round(n_cols * ratio)
        matrix = torch.randn(n_rows, n_cols)
        norm = torch.sqrt(torch.diag(torch.matmul(matrix, matrix.transpose(0, 1))))
        norm = torch.unsqueeze(norm, dim=0)
        matrix = matrix = matrix / norm.transpose(0, 1).repeat(1, n_cols)
        matrix_transpose = matrix.transpose(0, 1)

        return matrix, matrix_transpose

    def cal_r(self, x, y, w, h, init=False):
        """
        用来计算AT(y-Ax)
        """
        y_res = torch.matmul(torch.nn.functional.unfold(x, kernel_size=32, stride=32).transpose(1, 2), self.sample.transpose(0, 1)) - y
        if init:
            r = torch.matmul(y_res, self.init.transpose(0, 1))
        else:
            r = torch.matmul(y_res, self.sample)
        r = torch.nn.functional.fold(r.transpose(1, 2), output_size=(w, h), kernel_size=32, stride=32)

        return r

    def forward(self, ori_x):

        _, _, w, h = ori_x.shape
        output = []

        #采样
        y = torch.matmul(torch.nn.functional.unfold(ori_x, kernel_size=32, stride=32).transpose(1, 2), self.sample.transpose(0, 1))

        #初始化
        x_init = torch.matmul(y, self.init.transpose(0, 1))
        x_init = torch.nn.functional.fold(x_init.transpose(1, 2), output_size=(w, h), kernel_size=32, stride=32)
        
        #第一次收集信息
        x = x_init.clone()
        u = x_init.clone()
        r = self.cal_r(x, y, w, h, init=True)
        self.ics.collect(x, u, r)
        res = self.cal_r(x, y, w, h)

        for i in range(self.iteration):
            
            u, du = self.ublocks[i](x, u, self.ics)    #本次迭代ublock
            self.ics.collect(u=u)
            self.ics.collect(du=du)                     #收集u和du
            x = self.xblocks[i](x, u, res, self.ics)    #本次迭代xblock
            r = self.cal_r(x, y, w, h, init=True)       #计算r
            res = self.cal_r(x, y, w, h)
            self.ics.collect(x=x, r=r)                  #收集x和r

        x = self.deblocker(x)                           #去块伪影

        output.append(x)
        self.ics.clear()
        return output

class XBlock(nn.Module):

    def __init__(self, n_channel, n_information):
        super(XBlock, self).__init__()

        self.p1 = nn.Parameter(torch.Tensor([0.1]))
        self.p2 = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, x, u, res, ics):

        x = x - self.p1 * res - self.p2 * (x - u)

        return x

class UBlock(nn.Module):

    def __init__(self, n_channel, n_information):
        super(UBlock, self).__init__()

        self.denoiser = Denoiser(n_channel)
        self.p1 = nn.Parameter(torch.Tensor([0.1]))
        self.p2 = nn.Parameter(torch.Tensor([0.1]))
        self.fuse = Fuse(n_information, n_channel)

    def forward(self, x, u, ics):
        
        info = self.fuse(ics.x_deque, ics.u_deque, ics.r_deque, ics.du_deque)
        du = self.denoiser(u, info)
        u = u - self.p1 * (u - du) - self.p2 * (x - u)

        return u, du

#降噪器，这里使用了sparse non-local

class Denoiser(nn.Module):

    def __init__(self, n_channel):
        super(Denoiser, self).__init__()

        self.nl = NonLocalSparseAttention()
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(n_channel, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, u, info):

        info = self.nl(info)
        res = self.conv3(self.relu(self.conv2(info)))
        info = self.relu(info + res)
        res2 = self.conv4(info)

        out = u + res2

        return out

#去块效应模块
class DeBlocker(nn.Module):

    def __init__(self, n_channel):
        super(DeBlocker, self).__init__()

        self.conv1 = nn.Conv2d(1, n_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_channel, n_channel, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(n_channel, 1, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x,):

        res = self.relu(self.conv1(x))
        res = self.relu(self.conv3(self.relu(self.conv2(res))))
        res = self.conv4(res)

        x = x + res

        return x

#信息收集策略
class ICS():

    def __init__(self, maxlen=3):

        self.x_deque = deque(maxlen=maxlen)
        self.u_deque = deque(maxlen=maxlen)
        self.r_deque = deque(maxlen=maxlen)
        self.du_deque = deque(maxlen=maxlen)

    def collect(self, x=None, u=None, r=None, du=None):

        if x is not None:
            self.x_deque.append(x)
        if u is not None:
            self.u_deque.append(u)
        if r is not None:
            self.r_deque.append(r)
        if du is not None:
            self.du_deque.append(du)

    def clear(self):

        self.x_deque.clear()
        self.u_deque.clear()
        self.r_deque.clear()
        self.du_deque.clear()

class Fuse(nn.Module):

    def __init__(self, n_information, n_channel):
        super(Fuse, self).__init__()

        self.conv = nn.Conv2d(n_information, n_channel, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x_deque, u_deque, r_deque, du_deque):

        info = self.relu(self.conv(torch.cat([*x_deque, *u_deque, *r_deque, *du_deque], 1)))

        return info

#具体可以看Image Super-Resolution with Non-Local Sparse Attention，代码是这篇文章的
#想要进一步理解的话可以看原论文或者Reformer
#因为这里有一个用随机高斯做的投影，因此如果不固定种子的话，每次测试的结果都会不一样
class NonLocalSparseAttention(nn.Module):
    def __init__( self, n_hashes=4, channels=32, k_size=3, reduction=4, chunk_size=144, conv=common.default_conv, res_scale=1):
        super(NonLocalSparseAttention,self).__init__()
        self.chunk_size = chunk_size#ÿ��chunk��144��Ԫ�ص�hashֵ
        self.n_hashes = n_hashes#hashֵΪ4ά
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = common.BasicBlock(conv, channels, channels//reduction, k_size, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channels, channels, 1, bn=False, act=None)
        # ��Ϊ�Ա� ��׼Non-local����
        # self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act = nn.PReLU())
        # self.conv_assembly = common.BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())
 
    def LSH(self, hash_buckets, x):
        #x: [N,H*W,C]
        N = x.shape[0]#batch size
        device = x.device
        
        #generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets//2) #[1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1) #[N, C, n_hashes, hash_buckets//2]
        
        #locality sensitive hashing [n hw c]*[N, C, n_hashes, hash_buckets//2]
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, H*W, hash_buckets//2]����channelά���ڵ��ˣ�hw�������Ӧ����������ת������Ӧ����������ͼ�е���Ͳ���
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1) #[N, n_hashes, H*W, hash_buckets]
        #ΪʲôҪ�������أ������������и��������Բο�
        # [42] Kengo Terasawa and Yuzuru Tanaka. Spherical lsh for approximate nearest neighbor search on unit hypersphere. In Workshop on Algorithms and Data Structures, pages 27�C38. Springer, 2007. 3
        #�ĸ�¼����Ӧ��orthoplex�龰
 
        #get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1) #[N, n_hashes, H*W, hash_buckets]->[N,n_hashes,H*W]���ÿ��hash bucket������ֵ��λ�� ��Ϊ��feature map���ص��hashֵ
        
        #add offsets to avoid hash codes overlapping between hash rounds ����һ��ƫ��������ֹhash code�ص�
        offsets = torch.arange(self.n_hashes, device=device) #���ɡ�0��1��2��3������
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1)) #��0��1*hb,3*hb,3*hb��  ��״�ǣ�1��4��1��
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,)) #[N,n_hashes(���ά�Ⱥ�offsetsһ��),H*W]->[N,n_hashes*H*W]
    
        return hash_codes 
    
    def add_adjacent_buckets(self, x):
        #����������ڰ����ڵ�bucket����
        x_extra_back = torch.cat([x[:,:,-1:, ...], x[:,:,:-1, ...]], dim=2)#�ѵ�����һ���Ƶ��˵�һ�е�λ�� �൱�������ƶ�һ��
        x_extra_forward = torch.cat([x[:,:,1:, ...], x[:,:,:1,...]], dim=2)#�ѵ�һ���Ƶ��˵�����һ�е�λ�� �൱�������ƶ�һ��
        return torch.cat([x, x_extra_back,x_extra_forward], dim=3)#�����������������еķ������ƴ��
        #�������ʮ������ؽ���i�� ��i-1��i+1�������һ������ ƴ������������
 
    def forward(self, input):
        
        N,_,H,W = input.shape
        x_embed = self.conv_match(input).view(N,-1,H*W).contiguous().permute(0,2,1)#channel�� ?4�� [N,h*w,c/4]
        y_embed = self.conv_assembly(input).view(N,-1,H*W).contiguous().permute(0,2,1)#channel��û�б� [N,h*w,c]
        #contiguous��viewֻ��������contiguous��variable�ϣ������view֮ǰ������transpose��permute�ȣ�����Ҫ����contiguous()������һ��contiguous copy��
        #���Ϊʲô������permute֮�����contigious�أ����Ǻܶ�
        L,C = x_embed.shape[-2:] #L��H*W����C��channel/4
 
        #number of hash buckets/hash bits �����ж��ٸ�Ͱ�� ���128��
        hash_buckets = min(L//self.chunk_size + (L//self.chunk_size)%2, 128)#����hash_buckets��bucket����������ż��
        
        #get assigned hash codes/bucket number         
        hash_codes = self.LSH(hash_buckets, x_embed) #[N,n_hashes*H*W]
        hash_codes = hash_codes.detach()#������̲���Ҫ���򴫲�
 
        #group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1) #[N,n_hashes*H*W] sort���������У�����ֵΪvalue-tensor��indice-tensor
        _, undo_sort = indices.sort(dim=-1) #undo_sort to recover original order 
        #���ﷵ�ص��� ��N,n_hashes*H*W����һ�η���ֵ��ԭ����hash_codes��ÿһ��ֵ���Ĵ�С����������������������������������а�˳�����еĽ�����ǿ��Ը������undo-sort�б�����ԭ��ԭʼ����������
        mod_indices = (indices % L) #now range from (0->H*W)
        x_embed_sorted = common.batched_index_select(x_embed, mod_indices) #[N,n_hashes*H*W,C]
        y_embed_sorted = common.batched_index_select(y_embed, mod_indices) #[N,n_hashes*H*W,C*4]
        # def batched_index_select(values, indices):
        #     last_dim = values.shape[-1]
        #     return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
        #None�����������������һά��������np.newaxis
 
        #pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L%self.chunk_size if L%self.chunk_size!=0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes,-1, C)) #[N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes,-1, C*self.reduction)) 
        if padding:
            pad_x = x_att_buckets[:,:,-padding:,:].clone()
            pad_y = y_att_buckets[:,:,-padding:,:].clone()
            x_att_buckets = torch.cat([x_att_buckets,pad_x],dim=2)
            y_att_buckets = torch.cat([y_att_buckets,pad_y],dim=2)#����󼸸���Ϊpad������
        
        x_att_buckets = torch.reshape(x_att_buckets,(N,self.n_hashes,-1,self.chunk_size,C)) #[N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets,(N,self.n_hashes,-1,self.chunk_size, C*self.reduction))
        
        x_match = F.normalize(x_att_buckets, p=2, dim=-1,eps=5e-5)#L2��һ��
        #[N, n_hashes, num_chunks, chunk_size, C]
 
        #allow attend to adjacent buckets
        #������We then apply the Non-Local (NL) operation within the bucket that the query pixel belongs to, or across adjacent buckets after sorting.
        #Ϊ�˿����������ڵ���
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)
        
        #unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets, x_match) #[N, n_hashes, num_chunks, chunk_size, chunk_size*3]
        
        #softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)#logsumexpʵ���������max������һ��ƽ������
        score = torch.exp(raw_score - bucket_score) #(after softmax)
        bucket_score = torch.reshape(bucket_score,[N,self.n_hashes,-1])
        
        #attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets) #[N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret,(N,self.n_hashes,-1,C*self.reduction))
        
        #if padded, then remove extra elements
        if padding:
            ret = ret[:,:,:-padding,:].clone()
            bucket_score = bucket_score[:,:,:-padding].clone()
         
        #recover the original order
        ret = torch.reshape(ret, (N, -1, C*self.reduction)) #[N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,)) #[N,n_hashes*H*W]
        ret = common.batched_index_select(ret, undo_sort)#[N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)#[N,n_hashes*H*W]
        
        #weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C*self.reduction)) #[N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score,dim=1)
        ret = torch.sum(ret * probs, dim=1)
        
        ret = ret.permute(0,2,1).view(N,-1,H,W).contiguous()*self.res_scale+input
        return ret
