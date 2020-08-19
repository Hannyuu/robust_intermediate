import torch
import torch.nn as nn
from convex_adversarial import dual_network
import itertools
from torch.autograd import Variable
BATCH_SIZE = 8

def convert_cnn_to_dense(layer_cnn, input_size, test_batch=0, test_threshold=1e-6):
    """ Transforms a CNN layer for a given input size to an equivalent dense linear layer.

    """
    x = [(i,) + l for i, l in enumerate(itertools.product(*[range(s) for s in input_size]))]

    with torch.no_grad():
        i = torch.LongTensor(x)
        v = torch.ones(len(i))

        input_tensor = torch.sparse.LongTensor(i.transpose(0, 1), v, torch.Size([len(i)] + list(input_size))).cuda()
        output_tensor = layer_cnn(input_tensor.to_dense())
        output_size = output_tensor.size()[1:]

        bias_mlp = layer_cnn.bias.repeat_interleave(int(np.prod(output_size[-2:])))
        weight_mlp = (output_tensor.view(len(i), -1) - bias_mlp).transpose(0, 1)
        weight_mlp.to_sparse()

        if test_batch:
            # just for debugging purposes
            layer_mlp = nn.Linear(np.prod(input_size), np.prod(output_size))
            layer_mlp.weight.data = weight_mlp.to_dense()
            layer_mlp.bias.data = bias_mlp
            x = torch.randn([test_batch] + input_size)
            output_cnn = layer_cnn(x).view(test_batch, -1)
            output_mlp = layer_mlp(x.view(test_batch, -1))
            assert ((output_cnn - output_mlp).abs() < test_threshold).all()

        return weight_mlp, bias_mlp


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(2), -1)


def convert_cnn_net_to_fc_net(input_size, model, cuda=True):
    newmodel = torch.nn.Sequential()
    name = 0
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Conv2d):
            w, b = convert_cnn_to_dense(layer, input_size)
            newlayer = nn.Linear(w.shape[1], w.shape[0])
            newlayer.weight.data = w
            newlayer.bias.data = b
            newmodel.add_module(str(name), newlayer)
            input_size = layer(torch.ones(input_size).unsqueeze(0).cuda()).shape[1:]
            name += 1
        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.ReLU):
            newmodel.add_module(str(name), layer)
            name += 1
    print(newmodel.eval())
    return newmodel.cuda() if cuda else newmodel


def crownfirstbound(model, eps, inputdata, crown=True):
    layer = model[-1]
    assert isinstance(layer, nn.Linear)
    Ax0 = layer(inputdata)
    UB_new = Ax0 + eps * torch.max(abs(layer.weight), dim=1)[0]
    LB_new = Ax0 - eps * torch.max(abs(layer.weight), dim=1)[0]
    return LB_new, UB_new


def compute_slope(UB, LB, neuron_states, activation="Relu"):
    ## 0 ---> a_u, 1 ---> b_u,
    slope_intercepte = UB.new_zeros((4, UB.shape[0]))
    # +, a_u = 1, a_L = 1
    slope_intercepte[0, neuron_states == 1] = 1
    slope_intercepte[2, neuron_states == 1] = 1
    # +-, a_u = a, a_l = a
    slope_intercepte[0, neuron_states == 0] = UB[neuron_states == 0] / (UB[neuron_states == 0] - LB[neuron_states == 0])
    slope_intercepte[1, neuron_states == 0] = -LB[neuron_states == 0]
    slope_intercepte[2, neuron_states == 0] = UB[neuron_states == 0] / (UB[neuron_states == 0] - LB[neuron_states == 0])
    # pdb.set_trace()
    return slope_intercepte


def crown_intermediate_bound(model, nlayer, preReLU_UB, preReLU_LB, neuron_states, x0, eps, activation, slope_inter):
    new_slope_inter = compute_slope(preReLU_UB[-1], preReLU_LB[-1], neuron_states[-1], activation)
    slope_inter.append(new_slope_inter)
    out = model[nlayer].bias.shape[0]
    A = torch.eye(out).cuda()
    B = torch.eye(out).cuda()
    ##（10，10）
    j = len(slope_inter) - 1
    ## A ---- upper matrix , B ----- lower matrix
    f_u = A.matmul(model[nlayer].bias.view(out))
    f_l = B.matmul(model[nlayer].bias.view(out))
    for i in range(nlayer, 0, -1):
        if isinstance(model[i], nn.ReLU):
            continue
        AW = torch.matmul(A, model[i].weight)  # (10,1024)
        BW = torch.matmul(B, model[i].weight)
        # （10，1024）
        U_p = AW >= 0
        U_n = AW < 0
        l_p = BW >= 0
        l_n = BW < 0
        lam = slope_inter[j][0] * U_p + slope_inter[j][2] * U_n  # (10,1024)
        delta = slope_inter[j][1] * U_p + slope_inter[j][3] * U_n  # (1024,10)
        A = AW * lam  # (10,1024)
        temp = delta + model[i - 2].bias.expand(delta.shape)
        f_u = f_u + A.matmul(temp.t()).diag()

        omega = slope_inter[j][2] * l_p + slope_inter[j][0] * l_n
        theta = slope_inter[j][3] * l_p + slope_inter[j][1] * l_n
        B = BW * omega
        temp = theta + model[i - 2].bias.expand(theta.shape)
        f_l = f_l + B.matmul(temp.t()).diag()
        # print("A.shape:{}, B.shape: {} , AW.shape: {} , BW.shape: {}".format(A.shape,B.shape,AW.shape,BW.shape))
        j = j - 1
    i = i - 1
    AW = torch.matmul(A, model[i].weight)
    BW = torch.matmul(B, model[i].weight)
    # step 6: bounding A0 * x
    f_l = f_l + BW.matmul(x0.t()) - eps * torch.norm(BW, p=1, dim=1)
    f_u = f_u + AW.matmul(x0.t()) + eps * torch.norm(AW, p=1, dim=1)
    return f_u.detach(), f_l.detach()


def compute_worst_bound(model, x0, numlayer, eps=0.1,
                        untargeted=False, use_quad=False, activation="relu"):
    ## model: in pytorch, a dense neural network
    ## x0 : the flatten input,e.g (1,28,28)--> (784)
    ## numlay: the number of layers in the NN
    ## eps :verification epsilon
    neuron_states = []
    preReLU_UB = []
    preReLU_LB = []
    slope_inter = []
    output = []
    for num in range(numlayer):
        ## get naive bound for first layer
        if num == 0:
            LB, UB = crownfirstbound(model[:num + 1], eps, x0)
        elif isinstance(model[num], torch.nn.ReLU):
            continue
        else:
            UB, LB = crown_intermediate_bound(model, num, preReLU_UB, preReLU_LB, neuron_states, x0, eps, activation,
                                              slope_inter)
        preReLU_UB.append(UB)
        preReLU_LB.append(LB)
        output.append((LB.detach(), UB.detach()))
        # apply ReLU here manually (only used for computing neuron states)
        I = torch.zeros(UB.shape)
        # pdb.set_trace()
        I[(LB > 0)] = 1
        I[(LB < 0) * (UB >= 0)] = 0
        I[UB < 0] = -1
        neuron_states.append(I)
    return output

def replace_bound(inputsize, model,dataloader,eps = 0.001,pgd=False,verbose = 125):
    fcmodel = convert_cnn_net_to_fc_net(inputsize, model)
    err = 0
    aferr = 0
    imp = 0
    err_pgd = 0
    aferr_pgd = 0
    niters = 10
    alpha = 2/255
    for i,(X,Y) in enumerate(dataloader):
        if i > 20:
            break
        X = Variable(X.cuda())
        Y = Variable(Y.cuda())
        if pgd:
            X_pgd = X.clone().detach()
            X_pgd = Variable(X_pgd.cuda())
            for j in range(niters):
                X_pgd.requires_grad_()
                opt = optim.Adam([X_pgd], lr=1e-3)
                opt.zero_grad()
                loss = nn.CrossEntropyLoss()(model(X_pgd), Y)
                loss.backward()
                eta = alpha*X_pgd.grad.data.sign()
                X_pgd = X_pgd + eta

    # adjust to be within [-epsilon, epsilon]
                eta = torch.clamp(X_pgd - X, -eps, eps)
    # ensure valid pixel range
                X_pgd = torch.clamp(X + eta, 0, 1.0).detach()
            with torch.no_grad():
                f_pgd,dualnet_pgd= dual_network.RobustBounds(model,epsilon=eps,norm_type='l1')(X_pgd,Y)
                err_pgd = err_pgd + (f_pgd.max(1)[1] != Y).sum().data
                for index,item in enumerate(X_pgd):
                    crownout = compute_worst_bound(fcmodel, torch.flatten(item), len(fcmodel), eps, untargeted = False,use_quad = False, activation = "relu")
                    dualnet_pgd.dual_net[2].zl[index] = crownout[0][0].view(dualnet_pgd.dual_net[2].zl[0].shape)
                    dualnet_pgd.dual_net[4].zl[index] = crownout[1][0].view(dualnet_pgd.dual_net[4].zl[0].shape)
                    dualnet_pgd.dual_net[7].zl[index] = crownout[2][0].view(dualnet_pgd.dual_net[7].zl[0].shape)
                    dualnet_pgd.dual_net[2].zu[index] = crownout[0][1].view(dualnet_pgd.dual_net[2].zl[0].shape)
                    dualnet_pgd.dual_net[4].zu[index] = crownout[1][1].view(dualnet_pgd.dual_net[4].zl[0].shape)
                    dualnet_pgd.dual_net[7].zu[index] = crownout[2][1].view(dualnet_pgd.dual_net[7].zl[0].shape)
                num_classes = 10
                c = Variable(torch.eye(num_classes).type_as(X)[Y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
                fnew = -dualnet_pgd(c)
                aferr_pgd = aferr + (fnew.max(1)[1] != Y).sum().data
                if i%verbose == 0:
                    print("PGD error:",err_pgd,aferr_pgd)
        with torch.no_grad():

            f,dualnet= dual_network.RobustBounds(model,epsilon=eps,norm_type='l1')(X,Y)
            err = err + (f.max(1)[1] != Y).sum().data
            for index,item in enumerate(X):
                crownout = compute_worst_bound(fcmodel, torch.flatten(item), len(fcmodel), eps, untargeted = False,use_quad = False, activation = "relu")
                dualnet.dual_net[2].zl[index] = crownout[0][0].view(dualnet.dual_net[2].zl[0].shape)
                dualnet.dual_net[4].zl[index] = crownout[1][0].view(dualnet.dual_net[4].zl[0].shape)
                dualnet.dual_net[7].zl[index] = crownout[2][0].view(dualnet.dual_net[7].zl[0].shape)
                dualnet.dual_net[2].zu[index] = crownout[0][1].view(dualnet.dual_net[2].zl[0].shape)
                dualnet.dual_net[4].zu[index] = crownout[1][1].view(dualnet.dual_net[4].zl[0].shape)
                dualnet.dual_net[7].zu[index] = crownout[2][1].view(dualnet.dual_net[7].zl[0].shape)
            num_classes = 10
            c = Variable(torch.eye(num_classes).type_as(X)[Y].unsqueeze(1) - torch.eye(num_classes).type_as(X).unsqueeze(0))
            fnew = -dualnet(c)
            aferr = aferr + (fnew.max(1)[1] != Y).sum().data
            imp = imp + (f-fnew).sum().data
            if i%verbose == 0:
                print((f-fnew).mean().data,err,aferr)
    return imp/(float(i+1)*BATCH_SIZE),err/(float(i+1)*BATCH_SIZE),aferr/(float(i+1)*BATCH_SIZE),err_pgd/(float(i+1)*BATCH_SIZE),aferr_pgd/(float(i+1)*BATCH_SIZE)