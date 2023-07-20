# -*- coding: utf-8 -*
"""
在ErnieHierarClassification的基础上加入了local的表示
paddle代码里应该是觉得ernire抽取的特征足够了
globallayer对应的是local和global的主要代码，body里的代码主要是添加惩罚项
"""
import collections
import logging
from collections import OrderedDict
import codecs as cs
 
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from paddle import fluid
from wenxin.common.register import RegisterSet
from wenxin.common.rule import InstanceName
from wenxin.models.model import Model
from wenxin.modules.ernie import ErnieModel, ErnieConfig
import wenxin.metrics.metrics as metrics
from wenxin.data.util_helper import get_hierar_relations_array
from wenxin.models.ernie_hierar_label_classification import ErnieHierarLabelClassification
 
 
@RegisterSet.models.register
class SelfErnieHierarLabelClassificationLocal(ErnieHierarLabelClassification):
    """
        SelfErnieHierarLabelClassificationLocal
    """
     
    def __init__(self, model_params):
        ErnieHierarLabelClassification.__init__(self, model_params)
        self.hierar_relations = get_hierar_relations_array(model_params)
        self.global2local = [0, 5, 26]
 
    def forward(self, fields_dict, phase):
        """前向计算组网部分包括loss值的计算,必须由子类实现
        :param: fields_dict: 序列化好的id
        :param: phase: 当前调用的阶段，如训练、预测，不同的阶段组网可以不一样
        :return: 一个dict数据，存放TARGET_FEED_NAMES, TARGET_PREDICTS, PREDICT_RESULT,LABEL,LOSS等所有你希望获取的数据
        """
        fields_dict = self.fields_process(fields_dict, phase)
        instance_text_a = fields_dict["text_a"]
        record_id_text_a = instance_text_a[InstanceName.RECORD_ID]
        text_a_src = record_id_text_a[InstanceName.SRC_IDS]
        text_a_pos = record_id_text_a[InstanceName.POS_IDS]
        text_a_sent = record_id_text_a[InstanceName.SENTENCE_IDS]
        text_a_mask = record_id_text_a[InstanceName.MASK_IDS]
        text_a_task = record_id_text_a[InstanceName.TASK_IDS]
 
        instance_label = fields_dict["label"]
        record_id_label = instance_label[InstanceName.RECORD_ID]
        label = record_id_label[InstanceName.SRC_IDS]
 
        qid = None
        if "qid" in fields_dict.keys():
            instance_qid = fields_dict["qid"]
            record_id_qid = instance_qid[InstanceName.RECORD_ID]
            qid = record_id_qid[InstanceName.SRC_IDS]
 
        emb_dict = self.make_embedding(fields_dict, phase)
        emb_text_a = emb_dict["text_a"]
        emb_size = emb_text_a.shape[-1]
        self.hierar_depth = [0, emb_size, emb_size]
 
        num_labels = self.model_params.get("num_labels")
        hierar_penalty = self.model_params.get("hierar_penalty", 0.000001)
 
        # 添加global_local_re方法得到local和global的表征
        def global_local_re(emb_text_a):
            """
            global_layer是全局的表示
            local_layer是局部的表示
            """
            global_layer = emb_text_a
            emb_size = emb_text_a.shape[-1]
            local_layer_outputs = []
            for i in range(1, len(self.hierar_depth)):
                local_layer = fluid.layers.fc(input=global_layer, size=self.hierar_depth[i])
                local_layer_output = fluid.layers.fc(input=local_layer, size=self.global2local[i])
                local_layer_outputs.append(local_layer_output)
                if i < len(self.hierar_depth):
                    global_layer = fluid.layers.concat(input=[local_layer, emb_text_a], axis=1)
                else:
                    global_layer = local_layer
            local_layer_output_concat = fluid.layers.concat(input=local_layer_outputs, axis=1)
            return global_layer, local_layer_output_concat
         
        global_layer, local_layer_output_concat = global_local_re(emb_text_a)
 
        # 将global的表征送入fc层
        cls_feats = fluid.layers.dropout(
            x=global_layer,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
 
        logits = fluid.layers.fc(
            input=cls_feats,
            size=num_labels,
            param_attr=fluid.ParamAttr(
                name="_cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="_cls_out_b",
                initializer=fluid.initializer.Constant(0.)))
 
        # 最后将global和local拼接起来
        logits = 0.5 * logits + 0.5 * local_layer_output_concat
        probs = fluid.layers.sigmoid(logits)
 
        w = fluid.default_main_program().global_block().var("_cls_out_w")
        w_T = fluid.layers.transpose(w, perm=[1, 0])
 
        hierarchy_relation = fluid.layers.assign(self.hierar_relations)
 
        label_sum = fluid.layers.reduce_sum(label, dim=0)
        limit = fluid.layers.assign(np.array([0], dtype='float32'))
        label_flag = fluid.layers.greater_than(x=label_sum, y=limit)
        label_cast = fluid.layers.cast(label_flag, dtype="float32")
 
        label_cast_reshape = fluid.layers.reshape(x=label_cast, shape=[-1, 1])
        label_cast_expand = fluid.layers.expand(label_cast_reshape, expand_times=[1, num_labels])
        hierar_batch = fluid.layers.elementwise_mul(label_cast_expand, hierarchy_relation)
 
        def cond(i, loop_len, loss_norm):
            """
            循环条件
            :param i: 循环计数器
            :param loop_len: 循环次数
            :param loss_norm: 待获取的loss_norm值
            :return: 循环条件
            """
            return fluid.layers.less_than(x=i, y=loop_len)  # 循环条件
 
        def body(i, loop_len, loss_norm):
            """
            训练执行的结构体
            :param i: 循环计数器
            :param loop_len: 循环次数
            :param loss_norm: 待获取的loss_norm值
            :return: loop_vars
            """
            hierar_slice = fluid.layers.slice(hierar_batch, axes=[0], starts=i,
                                              ends=fluid.layers.increment(x=i, value=1, in_place=False))
            hierar_slice_T = fluid.layers.transpose(hierar_slice, perm=[1, 0])
            # 由于global每层都要和embedding拼接，所以expand时候的emb的纬度需要×2
            hierar_slice_expand = fluid.layers.expand(hierar_slice_T, expand_times=[1, emb_size * 2])
            child_params = fluid.layers.elementwise_mul(hierar_slice_expand, w_T)
            parent_param = fluid.layers.slice(w_T, axes=[0], starts=i,
                                              ends=fluid.layers.increment(x=i, value=1, in_place=False))
            parent_param_expand = fluid.layers.expand(parent_param, expand_times=[num_labels, 1])
            parent_params = fluid.layers.elementwise_mul(hierar_slice_expand, parent_param_expand)
            param_diff = fluid.layers.elementwise_sub(parent_params, child_params)
            param_diff_l2 = fluid.layers.reduce_sum(fluid.layers.square(param_diff)) * 0.5
            param_diff_result = fluid.layers.slice(label_sum, axes=[0], starts=i,
                                                   ends=fluid.layers.increment(x=i, value=1,
                                                                               in_place=False)) * param_diff_l2
            loss_norm = fluid.layers.elementwise_add(loss_norm, param_diff_result)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            return [i, loop_len, loss_norm]
 
        loss_norm = fluid.layers.assign(np.array([0], dtype='float32'))
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)  # 循环计数器
        loop_len = fluid.layers.fill_constant(shape=[1], dtype='int32', value=num_labels)  # 循环次数
        i, loop_len, loss_norm = fluid.layers.while_loop(cond, body, [i, loop_len, loss_norm])
 
        if phase == InstanceName.SAVE_INFERENCE:
            """保存模型时需要的入参：表示预测时最终输出的结果"""
            target_predict_list = [probs]
            """保存模型时需要的入参：表示模型预测时需要输入的变量名称和顺序"""
            target_feed_name_list = [text_a_src.name, text_a_pos.name, text_a_sent.name,
                                     text_a_mask.name]
            emb_params = self.model_params.get("embedding")
            ernie_config = ErnieConfig(emb_params.get("config_path"))
            if ernie_config.get('use_task_id', False):
                target_feed_name_list.append(text_a_task.name)
 
            forward_return_dict = {
                InstanceName.TARGET_FEED_NAMES: target_feed_name_list,
                InstanceName.TARGET_PREDICTS: target_predict_list
            }
            return forward_return_dict
 
        ce_loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=logits, label=label)
        ce_loss_with_hierar = fluid.layers.elementwise_add(ce_loss, hierar_penalty * loss_norm)
 
        total_sum = fluid.layers.reduce_sum(ce_loss_with_hierar, dim=-1)
        loss = fluid.layers.mean(x=total_sum) / num_labels if num_labels > 0 else fluid.layers.mean(x=total_sum)
 
        """PREDICT_RESULT,LABEL,LOSS 是关键字，必须要赋值并返回"""
 
        forward_return_dict = {
            InstanceName.PREDICT_RESULT: probs,
            InstanceName.LABEL: label,
            InstanceName.LOSS: loss,
        }
        if qid:
            forward_return_dict["qid"] = qid
        return forward_return_dict
