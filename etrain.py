import torch
from torch.autograd import Variable
import os

from Eval.eval import SOD_Eval

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from datetime import datetime
# from net.bgnet import Net
# from net.baseline import Net
from net.baseedgeRFEM import Net
# from net.bgnet_RFEM import Net
from utils.tdataloader import get_loader
from utils.utils import clip_gradient, AvgMeter, poly_lr
import torch.nn.functional as F
import numpy as np

file = open("log/BGNet.txt", "a")


# torch.manual_seed(2021)
# torch.cuda.manual_seed(2021)
# np.random.seed(2021)
# torch.backends.cudnn.benchmark = True

#该函数计算了一个综合损失函数，结合了加权二值交叉熵损失和加权交并比损失，以更好地处理图像分割任务中的边界区域。这个综合损失函数有助于模型更准确地分割图像中的目标物体，特别是在边界区域。
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask) # 计算权重 weit，它强调了边界区域的损失，F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)：对 mask 进行平均池化操作，池化窗口大小为 31，步幅为 1，填充为 15。
                                                                                               #torch.abs(...) - mask：计算池化结果与原始掩码 mask 的差的绝对值。
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')  # 计算加权的二值交叉熵损失 (Binary Cross Entropy Loss)
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)   # 使用 sigmoid 函数将预测值转换为概率
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()  # 返回加权的二值交叉熵损失和加权的交并比损失的平均值

#计算Dice损失，Dice损失衡量了预测的分割掩码与真实分割掩码之间的相似度。
#Dice损失适用于不平衡的分割任务，因为它关注交集与并集的比例，而不仅仅是逐像素的差异。通过使用平滑因子，函数避免了除零错误，使得损失计算更加稳定。
def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    #将预测值、目标值和有效掩码展平成二维张量，其中每行是一个样本的所有像素点。
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

#用于计算预测值和目标值之间的平均交并比 (Intersection over Union, IoU)。IoU 是衡量图像分割任务中预测分割结果与真实分割结果相似度的常用指标。
#IoU 损失 (1 - IoU) 用于优化模型，使其预测结果更接近真实分割结果
def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (modeltest-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


bce_loss = torch.nn.BCEWithLogitsLoss()
iou_loss = IOU(size_average=True)

#在每个训练周期开始时，模型被设置为训练模式。
#在每个批次中，数据被加载，损失被计算并记录，梯度被反向传播，参数被更新。
#每 60 步或在最后一步，打印当前的平均损失值。
def train(train_loader, model, optimizer, epoch):
    model.train()
    # 初始化记录损失的AvgMeter对象
    loss_record3, loss_record2, loss_record1, loss_recorde = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, edges = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        # ---- forward ----
        # lateral_map_3, lateral_map_2, lateral_map_1, edge_map = model(images)

        # lateral_map=model(images)
        sal, sal_sig, oe = model(images) #修改损失
        # sal, sal_sig = model(images)
        # ---- loss function ----计算了不同层输出的损失并将它们结合起来作为最终的损失值
        # loss3 = structure_loss(lateral_map_3, gts)    #lateral_map_3：模型的第三层侧向输出（可能是特征图）。# gts：真实标签（Ground Truth），即目标分割掩码。 # structure_loss：计算结构损失的函数，用于衡量预测值与真实值之间的差异。
        # loss2 = structure_loss(lateral_map_2, gts)
        # loss1 = structure_loss(lateral_map_1, gts)
        # losse = dice_loss(edge_map, edges)
        # loss = loss3 + loss2 + loss1 + 3*losse
        # loss=structure_loss(lateral_map,gts)
        # 修改损失
        loss_bce=bce_loss(sal, gts)
        loss_iou=iou_loss(sal_sig, gts)
        loss_edge=dice_loss(oe,edges)
        loss = bce_loss(sal, gts) + iou_loss(sal_sig, gts) + dice_loss(oe, edges)
        # loss = bce_loss(sal, gts) + iou_loss(sal_sig, gts) #没有边缘输入
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        # ---- recording loss ----
        # loss_record3.update(loss3.data, opt.batchsize)
        # loss_record2.update(loss2.data, opt.batchsize)
        # loss_record1.update(loss1.data, opt.batchsize)
        # loss_recorde.update(losse.data, opt.batchsize)
        # ---- train visualization ----
        if i % 60 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))
            file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                       '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}]\n'.
                       format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record3.avg, loss_record2.avg, loss_record1.avg, loss_recorde.avg))

        #修改损失之后
        # if i % 60 == 0 or i == total_step:
        #     print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
        #           '[loss: {:.4f}], [bce_loss: {:.4f}], [iou_loss: {:.4f}], [edge_loss: {:,.4f}]'.
        #           format(datetime.now(), epoch, opt.epoch, i, total_step,
        #                  loss.data, loss_bce.data, loss_iou.data, loss_edge.data))
        #     file.write('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
        #                '[loss: {:.4f}], [bce_loss: {:.4f}], [iou_loss: {:.4f}], [edge_loss: {:,.4f}]'.
        #                format(datetime.now(), epoch, opt.epoch, i, total_step,
        #                       loss.data, loss_bce.data, loss_iou.data, loss_edge.data))
    # 保存模型
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if epoch % 1 == 0:
        torch.save(model.state_dict(), save_path + 'BGNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'BGNet-%d.pth' % epoch)
        file.write('[Saving Snapshot:]' + save_path + 'BGNet-%d.pth' % epoch + '\n')
    # evaluate验证
    if (epoch + 1) % opt.val_interval == 0:
        SOD_Eval(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument('--epoch', type=int,
                        default=60, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--train_path', type=str,
                        default='./data/ORS-4199_aug/train', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='BGNet')
    parser.add_argument('--val_interval', type=int, default=1, help='validation interval')  # 多少轮验证一次
    opt = parser.parse_args()

    # ---- build models ----
    # 构建模型并移动到GPU
    model = Net().cuda()
    # 定义优化器
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    # 获取训练数据
    image_root = '{}/image/'.format(opt.train_path)
    gt_root = '{}/GT/'.format(opt.train_path)
    edge_root = '{}/edge/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    print("Start Training")

    for epoch in range(opt.epoch):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch)

    file.close()
