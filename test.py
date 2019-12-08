import os
import argparse
import glob
import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expresion,desired_expresion):
        img = cv_utils.read_cv2_img(img_path )
        morphed_img, concate_face = self._img_morph(img, expresion,desired_expresion)
        #concate_img = self._img_concate(img, expresion,desired_expresion)

        output_name = '%s_out.png' % os.path.basename(img_path)
        self._save_img(morphed_img, output_name)
        self._save_img(concate_face, "concate_" +  output_name)

    def _img_morph(self, img, expresion,desired_expresion):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face = self._morph_face(face, expresion, desired_expresion)
        concat_face = self._concate_face(face, expresion, desired_expresion)
        return (morphed_face, concat_face)

    def _morph_face(self, face, expresion,desired_expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/5.0), 0)
        desired_expresion = torch.unsqueeze(torch.from_numpy(desired_expresion/5.0), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': desired_expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['concat']

    
    def _concate_face(self, face, expresion,desired_expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/5.0), 0)
        desired_expresion = torch.unsqueeze(torch.from_numpy(desired_expresion/5.0), 0)
        import utils.util as util
        im_real_face = util.tensor2im(face.data)
        faces_list = [im_real_face] #.float().numpy()
        for idx in range(0, 6):
            cur_alpha = (idx + 1.) / 6.0
            cur_tar_aus = cur_alpha * desired_expresion + (1 - cur_alpha) * expresion
            test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': cur_tar_aus, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
            self._model.set_input(test_batch)
            imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
            # cur_gen_faces = self.test_model.fake_img.cpu().float().numpy()
            faces_list.append(imgs['fake_imgs_masked'])
        return np.concatenate(faces_list, 1)

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)


def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path
    conds_filepath = os.path.join(opt.data_dir, opt.aus_file)
    print(conds_filepath)

    with open(conds_filepath, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        aus = u.load()

    print(opt.test_id)
    print(aus[opt.test_id])

    expression = np.array(aus[opt.test_id])
    # desired_expression = np.random.uniform(0, 1, opt.cond_nc)
    desired_expression = np.array(aus[opt.test_id_desired])

    morph.morph_file(image_path, expression,desired_expression)


def loop_main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    testids = np.genfromtxt('./sample_dataset/test_ids.csv', dtype=str)
    img_dir = './sample_dataset/imgs/'
    for i in testids:
        image_path = img_dir + i
        expression = np.random.uniform(0, 1, opt.cond_nc)
        morph.morph_file(image_path, expression)

    


if __name__ == '__main__':
    main()
    ## loop_main()
    