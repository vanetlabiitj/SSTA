from pygsp import graphs, filters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def createGWCBlock(adj_mx):
    blocks = [STWNBlock(adj_mx, 2, 4, 4), STWNBlock(adj_mx, 4, 8, 4), STWNBlock(adj_mx, 8, 1, 4)]
    return blocks


def create_adj_kernel(N, size):
    Adj_kernel = nn.ParameterList(
        [nn.Parameter(torch.FloatTensor(N, N)) for _ in range(size)])
    return Adj_kernel


def create_mlp_kernel(in_channel, out_channel, kernel_size):
    mlp_kernel = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))
    return mlp_kernel


# def exp_wavelet_kernels(Lamda, scale):
#     if not isinstance(Lamda, np.ndarray):
#         Lamda = Lamda.detach().cpu().numpy()

#     kernels = [np.exp(-Lamda[i] * scale) for i in range(len(Lamda))]
#     # kernels = [np.exp(-Lamda[i] * 1.0 / i) for i in range(len(Lamda))]
#     #     print('scale:',scale, kernels)
#     return kernels

def exp_wavelet_kernels(Lamda, scale):
    return torch.exp(-Lamda * scale)

def create_wnn_kernel_by_pygsp(adj_mx):
    G = graphs.Graph(adj_mx)
    print('{} nodes, {} edges'.format(G.N, G.Ne))
    print(G)
    G.compute_laplacian('normalized')
    G.compute_fourier_basis()
    print('G.U', G.U)
    # print('G.U*G.UT', G.U )
    G.set_coordinates('ring2D')
    G.plot()

# TODO: this could be replaced by Chebyshev Polynomials
# def create_wnn_kernel_matrix(norm_laplacian, scale):
#     U, Lamda, _ = torch.svd(norm_laplacian)
#     kernels = exp_wavelet_kernels(Lamda, scale)
#     # print(Lamda)
#     G = torch.from_numpy(np.diag(kernels))
#     Phi = np.matmul(np.matmul(U, G), U.t())
#     Phi_inv = torch.inverse(Phi)
#     # print('create_wnn_kernel_matrix: Phi:', Phi)
#     return Phi, Phi_inv

def create_wnn_kernel_matrix(norm_laplacian, scale):
    # norm_laplacian: (N, N) torch tensor, can be on CPU or CUDA
    # Use SVD (or eigh for symmetric laplacian)
    U, Lamda, _ = torch.svd(norm_laplacian)
    # Wavelet kernels in torch
    kernels = exp_wavelet_kernels(Lamda, scale)        # (N,)
    # Diagonal matrix of kernels
    G = torch.diag(kernels)                            # (N, N)
    # Phi = U G U^T
    Phi = U @ G @ U.transpose(0, 1)                    # all torch ops
    # Inverse
    Phi_inv = torch.inverse(Phi)

    return Phi, Phi_inv



def get_wavelet(kernel, scale, adj_mx, is_gpu):
    Phi, Phi_inv = create_wnn_kernel_matrix(adj_mx, scale)

    device = adj_mx.device  # ensures all follow adjacency matrix device

    # Move kernel to same device
    kernel = kernel.to(device)

    return Phi.mm(kernel.diag()).mm(Phi_inv)


class WaveletKernel(nn.Module):
    def __init__(self, adj_mx, is_gpu=False, scale=0.1):
        super(WaveletKernel, self).__init__()
        self.is_gpu = is_gpu
        self.adj_mx = adj_mx
        self.N = adj_mx.shape[0]
        self.scale = scale

        self.g = nn.Parameter(torch.ones(self.N))
        self.Phi, self.Phi_inv = create_wnn_kernel_matrix(self.adj_mx, self.scale)

        if is_gpu:
            self.g = nn.Parameter(torch.ones(self.N).cuda())
            self.Phi = self.Phi.cuda()
            self.Phi_inv = self.Phi_inv.cuda()
        g_diag = self.g.diag()
        self.k = self.Phi.mm(g_diag).mm(self.Phi_inv)
        self.weight_init()

    def forward(self, x):
        # batch, feature, N
        x = torch.einsum('bfn, np -> bfp', x, self.k).contiguous()
        return x

    def weight_init(self):
        # this could be replaced by Chebyshev Polynomials
        nn.init.uniform_(self.g)


# TODO: wavelet attention mechanism and trans attention
class STWNBlock(nn.Module):
    def __init__(self, adj_mx, in_channel, out_channel, wavelets_num, is_gpu=True):
        super(STWNBlock, self).__init__()
        self.is_gpu = is_gpu
        self.N = adj_mx.shape[0]
        self.adj_mx = adj_mx
        self.adj_mx_t = adj_mx.float()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = wavelets_num
        device = adj_mx.device

        scales = [0.1 + 0.1 * 2 * i for i in range(wavelets_num)]
        kernel_para = torch.ones(wavelets_num, self.N, device=device).float()
        if is_gpu:
            kernel_para = kernel_para
        self.kernels = nn.ParameterList([nn.Parameter(kernel_para[i]) for i in range(wavelets_num)])
        self.wavelets = torch.stack(
            [get_wavelet(self.kernels[i], scales[i], self.adj_mx, is_gpu) for i in range(wavelets_num)],
            dim=0).to(device)
        self.randw = nn.Parameter(torch.randn(self.N, self.N, device=device).float())
        self.krandw = nn.Parameter(torch.stack([torch.randn(self.N, self.N) for i in range(wavelets_num)], dim=0).float())
#         self.wavelets = nn.Parameter(torch.randn(self.N, self.N).float())
        #print('wavelets shape', self.wavelets.shape)
        #self.Gate = nn.Parameter(torch.FloatTensor(wavelets_num, device=device))
        self.Gate = nn.Parameter(torch.ones(wavelets_num, device=device))
        #self.Gate = nn.Parameter(torch.tensor(wavelets_num, dtype=torch.float32, device=device))
        self.SumOne = torch.ones(wavelets_num, device=device).float()
        self.upsampling = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))
        self.rnn = nn.GRU(input_size=32,
                          hidden_size=32,
                          num_layers=1,
                          batch_first=True)
        self.weight_init()


    def forward(self, x):
        """
        :param x: (batch, in_channel, N, sequence)
        :return: (batch, out_channel, N, sequence)
        """

        seq_len = x.shape[3]
        seqs = []
        B, F, N, T = x.shape

        x = x.transpose(1, 2)


        # real wavelet + K randw
        wavelets = self.wavelets * self.krandw
        x = torch.einsum('bnft, knm -> bkmft', x, wavelets)
        x = torch.einsum('bknft, k -> bnft', x, self.Gate)


        # GCN + GRU:
        x = x.transpose(2, 3).reshape(B * N, T, F)
        outputs, last_hidden = self.rnn(x, None)
        outputs = outputs.reshape(B, N, T, F).permute(0, 3, 1, 2)
        return outputs

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                print('init module with kaiming', m)
            elif isinstance(m, nn.ParameterList):
                for i in m:
                    nn.init.normal_(i, mean=0.0, std=0.001)
            else:
                print('ParameterList!Do nothing')
        nn.init.normal_(self.Gate, mean=0.0, std=0.001)
        # nn.init.kaiming_normal_(self.sampling)
        # nn.init.kaiming_normal_(self.rnn.weight, mode='fan_in')


class AttSTWNBlock(nn.Module):
    '''
    Attention STWNBlock
    '''

    def __init__(self, adj_mx, in_channel, out_channel, kernel_size, att_channel=32, bn=True, sampling=None,
                 is_gpu=False):
        super(AttSTWNBlock, self).__init__()
        self.is_gpu = is_gpu
        print('AttSTWNBlock, is_gpu', is_gpu)
        self.N = adj_mx.shape[0]
        self.adj_mx = adj_mx
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.att_channel = att_channel

        scales = [0.1 + 0.1 * 2 * i for i in range(kernel_size)]
        kernel_para = torch.ones(kernel_size, self.N).float()

        if is_gpu:
            kernel_para = kernel_para.cuda()

        self.kernels = nn.ParameterList([nn.Parameter(kernel_para[i]) for i in range(kernel_size)])
        self.wavelets = torch.stack(
            [get_wavelet(self.kernels[i], scales[i], self.adj_mx, is_gpu) for i in range(kernel_size)],
            dim=0)
        print('wavelets shape', self.wavelets.shape)
        if is_gpu:
            self.upsamplings = nn.Parameter(torch.FloatTensor(kernel_size, in_channel, out_channel).cuda())
        else:
            self.upsamplings = nn.Parameter(torch.FloatTensor(kernel_size, in_channel, out_channel))

        #         self.upsamplings = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channel, out_channel).cuda())
        #                                              for _ in range(self.kernel_size)])

        self.upsampling = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1))

        self.Att_W = nn.Parameter(torch.FloatTensor(self.out_channel, self.att_channel))
        self.Att_U = nn.Parameter(torch.FloatTensor(kernel_size, self.att_channel))

        self.weight_init()

    def forward(self, x):
        """
        :param x: (batch, in_channel, N, sequence)
        :return: (batch, out_channel, N, sequence)
        """
        # TODO: change to recursive
        sha = x.shape
        seq_len = sha[3]
        seqs = []
        if self.is_gpu:
            wavelets = self.wavelets.cuda()
        else:
            wavelets = self.wavelets

        #         x = self.upsampling(x)
        #         print('samp',self.upsampling.weight)

        # attention:

        #       # wavelet gated, sum directly:

        #print("torch and wavelet shape", x.shape, wavelets.shape)


        x = x.permute(0, 2, 1, 3)
        x = torch.einsum('bnft, knm -> bkmft', x, wavelets)
        #print("mutiscale graph convolution shape", x.shape)
        x = torch.einsum('bkmft, kfo -> bknft')
        x = torch.einsum('bknft, k -> bnft', x, self.Gate)


        a = torch.einsum('fs, bknf -> bkns', self.Att_W, xs)
        a = torch.einsum('bkns, ks -> bkn', xs, self.Att_U)

        for i in range(seq_len):
            xs = x[..., i].transpose(1, 2)
            # wavelet transform:
            # bnf, knn -> bknf
            xs = torch.einsum('bnf, ksn -> bknf', xs, wavelets)

            # in_channel to out_channel, bknf , kfs -> bkns
            xs = torch.einsum('bknf , kfo -> bkno', xs, self.upsamplings)

            mask = xs == float('-inf')
            xs = xs.data.masked_fill(mask, 0)
            # attention:

            # fs, bknf -> bkns
            # bkns, s -> bkn
            # bkn -> a = softmax(bkn, k)
            # bknf, bkn -> bknf
            a = torch.einsum('fs, bknf -> bkns', self.Att_W, xs)
            a = torch.einsum('bkns, ks -> bkn', xs, self.Att_U)
            a = norm(a, dim=1)
            a = F.softmax(a, dim=1)
            a = a.transpose(1, 2)

            # mock attention:
            #             xsshape = xs.shape
            #             a = torch.ones(xsshape[0], xsshape[2], xsshape[1]).float().cuda()

            # xs * attention
            out = torch.einsum('bnk, bkno -> bno', a, xs).transpose(1, 2)
            #             h = out
            seqs.append(out)

        # stack all sequences
        x = torch.stack(seqs, dim=3)
        return x

    def norm(tensor_data, dim=0):
        mu = tensor_data.mean(axis=dim, keepdim=True)
        std = tensor_data.std(axis=dim, keepdim=True)
        return (tensor_data - mu) / (std + 0.00005)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                print('init module with kaiming', m)
            elif isinstance(m, nn.ParameterList):
                for i in m:
                    #                     nn.init.kaiming_normal_(i.data, mode='fan_out')
                    nn.init.normal_(i, mean=0.0, std=0.001)
                    # nn.init.kaiming_normal_(i.weight.data, mode='fan_in')
                print('init parameterlist with kaiming', m)
            else:
                print('ParameterList!Do nothing', m)
        # nn.init.kaiming_normal_(self.upsamplings.data, mode='fan_in')
        #         nn.init.normal_(self.upsamplings, mean=0.0, std=0.001)
        for m in self.upsamplings:
            nn.init.kaiming_normal_(m.data, mode='fan_in')

        nn.init.normal_(self.Att_W, mean=0.0, std=0.001)
        nn.init.kaiming_normal_(self.Att_U.data, mode='fan_in')

class DecoderRNN(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, out_channel=1, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.out_channel = out_channel
        # RNN层
        self.rnn = nn.RNN(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        print("DecoderRNN out_channel: ", out_channel)
        self.l1 = nn.Conv1d(hidden_len, out_channel, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch, N, sequence, out_channel)
        """
        batch, channel, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, channel, seq_len)
        h = None
        x = x[:, :, seq_len - 1].unsqueeze(dim=1)
        seqs = []
        for _ in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = out.transpose(1, 2)
            out = self.l1(out)
            seqs.append(out)

        predict = torch.cat(seqs, dim=2)
        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict

class NewDecoderRNN(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, out_channel=1, num_layers=1):
        super(NewDecoderRNN, self).__init__()
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.out_channel = out_channel
        # RNN层
        self.rnn = nn.RNN(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        print("DecoderRNN out_channel: ", out_channel)
        self.l1 = nn.Conv1d(hidden_len, out_channel, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch, N, sequence, out_channel)
        """
        batch, channel, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, channel, seq_len).transpose(1, 2)
        # last seq:
        x = x[:,-1,:].unsqueeze(dim=1)
        h = None
        seqs = []
        for i in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = out[:,-1,:].unsqueeze(dim=1)
            each_seq = out.transpose(1, 2)
            each_seq = self.l1(each_seq)
            seqs.append(each_seq)

        predict = torch.cat(seqs, dim=2).squeeze()
        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict


class DecoderLSTM(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, num_layers=1):
        super(DecoderLSTM, self).__init__()
        print('init DecoderLSTM')
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        # RNN层
        self.rnn = nn.LSTM(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        self.l1 = nn.Conv1d(hidden_len, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch,N,sequence)
        """
        batch, _, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, seq_len, self.feature_len).transpose(1, 2)
        h = None
        x = x[:, :, seq_len - 1].unsqueeze(dim=1)
        seqs = []
        for _ in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = self.l1(out.transpose(1, 2))
            seqs.append(out)

        predict = torch.cat(seqs, dim=1)

        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict


class DecoderGRU(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, num_layers=1):
        super(DecoderGRU, self).__init__()
        print('init DecoderGRU')
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        # RNN层
        self.rnn = nn.GRU(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        self.l1 = nn.Conv1d(hidden_len, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, feature, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch,N,sequence)
        """
        batch, _, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, seq_len, self.feature_len).transpose(1, 2)
        h = None
        x = x[:, :, seq_len - 1].unsqueeze(dim=1)
        seqs = []
        for _ in range(self.predict_len):
            out, h = self.rnn(x, h)
            # out = (batch * N, seq_len, feature) to (batch * N, feature, seq_len)
            out = self.l1(out.transpose(1, 2))
            seqs.append(out)

        predict = torch.cat(seqs, dim=1)

        predict = predict.reshape(batch, N, self.predict_len)
        # predict = self.dropout(predict)
        return predict, predict

class EncoderRNN(nn.Module):
    def __init__(self, feature_len, hidden_len, predict_len, out_channel=1, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.feature_len = feature_len
        self.hidden_len = hidden_len
        self.num_layers = num_layers
        self.predict_len = predict_len
        self.out_channel = out_channel
        # RNN层
        self.rnn = nn.RNN(
            input_size=feature_len,  # feature len
            hidden_size=hidden_len,  # 隐藏记忆单元尺寸
            num_layers=num_layers,  # 层数
            batch_first=True  # 在喂入数据时,按照[batch,seq_len,feature_len]的格式
        )
        print("DecoderRNN out_channel: ", out_channel)
        self.l1 = nn.Conv1d(hidden_len, out_channel, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        # 对RNN层的参数做初始化
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

    def forward(self, x):
        """
        x = (batch, channel, N, sequence)
        需要转换成:
        x = (batch x N, sequence, feature)
        :return:输出out(batch, in_channel, N, sequence)
        """
        batch, channel, N, seq_len = x.shape
        x = x.transpose(1, 2).reshape(batch * N, channel, seq_len)
        h = None
        x = x.transpose(1, 2)
        seqs = []
        out, h = self.rnn(x, h)
        # batch*N, seq, feature.
        out = out.reshape(batch, N, seq_len, self.hidden_len)
        out = out[:,:,-1,:].unsqueeze(dim=2).permute(0, 3, 1, 2)
        return out

class FC(nn.Module):
    def __init__(self, in_channel, out_channel, N):
        super(FC, self).__init__()
        self.N = N
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.l1 = nn.Conv2d(in_channel, 16, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(16)
        self.l2 = nn.Conv2d(16, out_channel, kernel_size=(1, 1))

    def forward(self, x):
        """
        :param x: (batch, features, Nodes, sequence)
        :return: (batch, sequence, Nodes, features=1).squeeze() --> (batch, sequence, Nodes)
        """
        x = F.relu(self.bn(self.l1(x)))
        x = F.relu(self.l2(x))
        seq_len = x.shape[3]
        batch_size = x.shape[0]
        # outs = []
        # for i in range(self.N):
        #     node = x[:, :, i, :]
        #     out = F.relu(self.l(node))
        #     outs.append(out)
        x = x.reshape(batch_size, seq_len, self.N, self.out_channel).squeeze()
        # x = torch.cat(outs, dim=2).reshape(batch_size, seq_len, self.N, self.out_channel).squeeze()
        return x

class STWN(nn.Module):
    def __init__(self, device, num_nodes, adj_mx, input_dim =1, output_dim = 12, horizon = 12):
        super(STWN, self).__init__()
        self.N = adj_mx.shape[0]
        self.feature_len =  input_dim
        self.predict_len = 12
        self.rnn_layer_num = 2
        self.upsampling = nn.Conv2d(self.feature_len, 32, kernel_size=(1, 1))
        self.wavelets_num = 20
        self.att = False
        self.gcn_layer_num = 2
        self.seq_len = 1
        self.batch_size = 128
        self.num_nodes = num_nodes
        self.horizon = 12
        self.device = device
        #self.adj_mx = adj_mx.to(device)
        is_gpu = (device.type == "cuda")

        # add rnn encoder before gcn:
        # self.encoder = DecoderRNN(args.feature_len, 32, args.predict_len, num_layers=args.rnn_layer_num)
        self.encoder = EncoderRNN(self.feature_len, 32, self.predict_len, out_channel=self.predict_len, num_layers=self.rnn_layer_num)
        if self.att == True:
            print('using att')
            self.gwblocks = nn.ModuleList([AttSTWNBlock(adj_mx, self.feature_len, 32, self.wavelets_num, is_gpu=is_gpu)
                                           ])
            if self.gcn_layer_num > 1:
                for i in range(1, self.gcn_layer_num):
                    self.gwblocks.append(AttSTWNBlock(adj_mx, 32, 32, self.wavelets_num, is_gpu=is_gpu))
        else:
            print('no att', self.att)
            self.gwblocks = nn.ModuleList(
                [STWNBlock(adj_mx, 32, 32, self.wavelets_num, is_gpu=is_gpu)
                 ])
            if self.gcn_layer_num > 1:
                for i in range(1, self.gcn_layer_num):
                    self.gwblocks.append(STWNBlock(adj_mx, 32, 32, self.wavelets_num, is_gpu=is_gpu))
                print("gcn_layer_num: ", self.gcn_layer_num)
        # self.readout = STWNBlock(adj_mx, 4, 1, 4)
        # residual + input feature
        self.W_W = nn.Parameter(torch.FloatTensor(self.N, self.predict_len))
        self.D_W = nn.Parameter(torch.FloatTensor(self.N, self.predict_len))
        self.H_W = nn.Parameter(torch.FloatTensor(self.N, self.predict_len))

        self.decoder = NewDecoderRNN(32, 32, self.predict_len, num_layers=self.rnn_layer_num)
        # self.decoder = DecoderRNN(32, 32, self.predict_len, num_layers=args.rnn_layer_num)

        self.lout = nn.Conv1d(32, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.2)
        self.weight_init()

    def forward(self, x, A_hat, edges, edge_weights):
        """
        :param x: (batch, in_channel, N, sequence)
        """
        
        #print("shape of x", x.shape)
        
        x = x.permute(0,2,1,3)
        out, _ = self.forward_one(x)

        #out = out.transpose(1, 2)
        #print("shape of out", out.shape)

        return out


    def forward_one(self, x):
        # x = (B, F, N, T)
        # encoder:

        # lstm + GCN + fc:
        # x = self.encoder(x)

        # GCN + lstm:
        #x = x.permute(1, 3, 2, 0)

        residual = F.relu(self.upsampling(x))
        h = x
        #print("shape of residual and x", residual.shape, x.shape)
        for i in range(0, len(self.gwblocks)):
            h = residual + self.gwblocks[i](residual)

        # skip connection
        out = residual + h
        out = F.relu(out)

        # GCN + LSTM, decoder:
        out, h = self.decoder(out)

        # lstm + GCN + fc:
        # out = self.lout(out)

        # test without fc:
        # out = out.squeeze().transpose(1, 2)
        # out = self.fc(out)

        # out = B, N, T
        return out, out

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                print('STWN init module with kaiming', m)
            elif isinstance(m, nn.ParameterList):
                for i in m:
                    nn.init.normal_(i, mean=0.0, std=0.001)
                print('STWN init parameterlist with norm')
            else:
                print('STWN ParameterList!Do nothing')
        # nn.init.kaiming_normal_(self.rnn.weight, mode='fan_in')
        nn.init.normal_(self.W_W, mean=0.0, std=0.001)
        nn.init.normal_(self.D_W, mean=0.0, std=0.001)
        nn.init.normal_(self.H_W, mean=0.0, std=0.001)


 