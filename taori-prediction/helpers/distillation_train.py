import sys
import torch
import torch.nn.functional as F
from operator import itemgetter
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter


_teacher_results = dict()


class DistillationEpoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        
        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s
    
    def _update_loss_logs(self, logs, loss_meter, loss, name):
        if loss is not None:
            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
#             if self.loss.__name__:
#                 name = self.loss.__name__
            loss_logs = {name: loss_meter.mean}
            logs.update(loss_logs)

    def batch_update(self, patient_id, teacher_x, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        metrics_logs = dict()
        loss_meter = AverageValueMeter()
        student_loss_meter = AverageValueMeter()
        distillation_loss_meter = AverageValueMeter()

        classes_gt = None
        classes_pr = None
        
        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for patient_id, teacher_x, x, y in iterator:
                teacher_x, x, y = teacher_x.to(self.device), x.to(self.device), y.to(self.device)
                loss, student_loss, distillation_loss, y_pred = self.batch_update(patient_id, teacher_x, x, y)

                # update loss logs
                self._update_loss_logs(logs, loss_meter, loss, 'loss')
                self._update_loss_logs(logs, student_loss_meter, student_loss, 's-loss')
                self._update_loss_logs(logs, distillation_loss_meter, distillation_loss, 'd-loss')

                _, class_pr = torch.max(y_pred, dim=-1)
                _, class_gt = torch.max(y, dim=-1)
                if classes_gt is None:
                    classes_gt = class_gt
                else:
                    classes_gt = torch.cat((classes_gt, class_gt))
                if classes_pr is None:
                    classes_pr = class_pr
                else:
                    classes_pr = torch.cat((classes_pr, class_pr))
                
                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(classes_pr, classes_gt).cpu().detach().numpy()
                    metrics_logs[metric_fn.__name__] = metric_value
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainDistillationEpoch(DistillationEpoch):
    def __init__(self, teacher, student, loss, distillation_loss, metrics, optimizer,
                 temperature=1, alpha=0.1, distillation_loss_scale=1,
                 device='cpu', cache_teacher_results=True, feature_based_kd=False, verbose=True):
        self.teacher = teacher
        self.distillation_loss = distillation_loss
        super().__init__(
            model=student,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_loss_scale = distillation_loss_scale
        self.cache_teacher_results = cache_teacher_results
        self.feature_based_kd = feature_based_kd

    def _to_device(self):
        super()._to_device()
        self.teacher.to(self.device)
        self.distillation_loss.to(self.device)
        
    def on_epoch_start(self):
        self.model.train()
        self.teacher.eval()
        
    def batch_update(self, patient_id, teacher_x, x, y):
        if self.cache_teacher_results and all([p in _teacher_results for p in patient_id]):
            teacher_prediction = itemgetter(*patient_id)(_teacher_results)
            teacher_prediction = torch.cat([p.unsqueeze(0) for p in teacher_prediction])
        else:
            with torch.no_grad():
                teacher_prediction = self.teacher.forward(teacher_x)
            if self.cache_teacher_results:
                for i in range(0, len(patient_id)):
                    _teacher_results[patient_id[i]] = teacher_prediction[i]

        self.optimizer.zero_grad()
        if self.feature_based_kd:
            prediction, guided_layer = self.model.forward(x)
        else:
            prediction = self.model.forward(x)
        
        student_loss = self.loss(prediction, y)
        if self.feature_based_kd:
            distillation_loss = self.distillation_loss(guided_layer / self.temperature,
                                                       teacher_prediction / self.temperature
                                                      ) * self.temperature**2
        elif isinstance(self.distillation_loss, torch.nn.KLDivLoss):
            # pytorch KLDivLoss implementation requires student prediction in log softmax
            distillation_loss = self.distillation_loss(F.log_softmax(prediction / self.temperature, dim=1),
                                                       F.softmax(teacher_prediction / self.temperature, dim=1)
                                                      ) * self.temperature**2
        else:
            distillation_loss = self.distillation_loss(F.softmax(prediction / self.temperature, dim=1),
                                                       F.softmax(teacher_prediction / self.temperature, dim=1)
                                                      ) * self.temperature**2
        distillation_loss = distillation_loss * self.distillation_loss_scale
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        
        loss.backward()
        self.optimizer.step()
        return loss, student_loss, distillation_loss, prediction


class ValidDistillationEpoch(DistillationEpoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, patient_id, teacher_x, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, None, None, prediction
