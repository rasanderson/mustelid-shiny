# Copyright CNRS 2026

# simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr

# This software is a computer program whose purpose is to identify
# animal species in camera trap images.

#This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 

# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 

# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import sys
import os
import timm
import torch
from torch import tensor
import torch.nn as nn
from torchvision.transforms import InterpolationMode, transforms

# Important: The default image size for DINOv3 ViT-L is 256x256. However, the DeepFaune classifier has been fine-tuned (and deployed here)
# with a 224x224 image size to speed up computation. Using the finetuned model at different resolution should also work but as not been
# tested (except for 256x256, which gives the same accuracy as 224x224 on our large test)
CROP_SIZE = 224
BACKBONE = "vit_large_patch16_dinov3.lvd1689m"
DFPATH = os.path.abspath(os.path.dirname(__file__))

DFVIT_WEIGHTS = os.path.join(DFPATH,'deepfaune-vit_large_patch16_dinov3.lvd1689m.pt')
DFBIRD_WEIGHTS = os.path.join(DFPATH,'deepfaune-vit_large_patch16_dinov3.lvd1689m-bird_head.pt')

txt_animalclasses = {
    'fr': ['bison', 'blaireau', 'bouquetin', 'castor', 'cerf', 'chacal doré', 'chamois', 'chat', 'chevre',
           'chevreuil', 'chien', 'chien viverrin', 'daim', 'ecureuil', 'elan', 'equide', 'genette', 'glouton',
           'herisson', 'lagomorphe', 'loup', 'loutre', 'lynx', 'mangouste', 'marmotte', 'micromammifere', 'mouflon', 'mouton',
           'mustelide', 'oiseau', 'ours', 'porcepic', 'ragondin', 'rat musqué', 'raton laveur', 'renard',
           'renard arctique', 'renne', 'sanglier', 'vache'],
    'en': ['bison', 'badger', 'ibex', 'beaver', 'red deer', 'golden jackal', 'chamois', 'cat', 'goat',
           'roe deer', 'dog', 'raccoon dog', 'fallow deer', 'squirrel', 'moose', 'equid', 'genet',
           'wolverine', 'hedgehog', 'lagomorph', 'wolf', 'otter', 'lynx', 'mongoose', 'marmot', 'micromammal', 
           'mouflon', 'sheep', 'mustelid', 'bird', 'bear', 'porcupine', 'nutria', 'muskrat', 'raccoon',
           'fox', 'arctic fox', 'reindeer', 'wild boar', 'cow'],
    'it': ['bisonte', 'tasso', 'stambecco', 'castoro', 'cervo', 'sciacallo dorato', 'camoscio', 'gatto', 'capra',
           'capriolo', 'cane', 'cane procione', 'daino', 'scoiattolo', 'alce', 'equide', 'genetta', 'ghiottone',
           'riccio', 'lagomorfo', 'lupo', 'lontra', 'lince', 'mangusta', 'marmotta', 'micromammifero', 'muflone', 'pecora',
           'mustelide', 'uccello', 'orso', 'istrice', 'nutria', 'ondatra', 'procione', 'volpe',
           'volpe artica', 'renna', 'cinghiale', 'mucca'],      
    'de': ['Bison', 'Dachs', 'Steinbock', 'Biber', 'Rothirsch', 'Goldschakal', 'Gämse', 'Katze', 'Ziege',
           'Rehwild', 'Hund', 'Marderhund', 'Damwild', 'Eichhörnchen', 'Elch', 'Equide', 'Ginsterkatze',
           'Vielfraß', 'Igel', 'Lagomorpha', 'Wolf', 'Otter', 'Luchs', 'Manguste', 'Murmeltier', 'Kleinsäuger', 'Mufflon',
           'Schaf', 'Marder', 'Vogel', 'Bär', 'Stachelschwein', 'Nutria', 'Bisamratte', 'Waschbär', 'Fuchs',
           'Polarfuchs', 'Rentier', 'Wildschwein', 'Kuh'],
    'es': ['bisonte', 'tejón', 'cabra montés', 'castor', 'ciervo rojo', 'chacal dorado', 'rebeco', 'gato', 'cabra',
           'corzo', 'perro', 'perro mapache', 'gamo', 'ardilla', 'alce', 'équido', 'gineta', 'glotón', 'erizo', 'lagomorfo',
           'lobo', 'nutria', 'lince', 'mangosta', 'marmota', 'micromamífero', 'muflón', 'oveja', 'mustélido', 'ave', 'oso', 'puercoespín',
           'coipú', 'rata almizclera', 'mapache', 'zorro', 'zorro ártico', 'reno', 'jabalí', 'vaca'],    
    'no': ['bison', 'grevling', 'steinbukk', 'bever', 'hjort', 'gullsjakal', 'gemse', 'katt', 'geit',
           'rådyr', 'hund', 'mårhund', 'dåhjort', 'ekorn', 'elg', 'hestedyr', 'genett', 'jerv',
           'pinnsvin', 'hare', 'ulv', 'oter', 'gaupe', 'mangust', 'murmeldyr', 'småpattedyr', 'muflon', 'sau',
           'mårdyr', 'fugl', 'bjørn', 'piggsvin', 'beverrotte', 'bisamrotte', 'vaskebjørn', 'rev',
           'fjellrev', 'rein', 'villsvin', 'ku'],  
    'se': ['bison', 'grävling', 'stenbock', 'bäver', 'kronhjort', 'guldschakal', 'gems', 'katt', 'get',
           'rådyr', 'hund', 'mårdhund', 'dovhjort', 'ekorre', 'älg', 'hästdjur', 'genett', 'järv',
           'igelkott', 'hare', 'varg', 'utter', 'lo', 'mangust', 'murmeldjur', 'smådäggdjur', 'mufflonfår', 'får',
           'mårdjur', 'fågel', 'björn', 'piggsvin', 'sumpbäver', 'bisam', 'tvättbjörn', 'räv',
           'fjällräv', 'ren', 'vildsvin', 'ko'],
    'pt': ['bisão', 'texugo', 'íbex', 'castor', 'veado', 'chacal-dourado', 'camurça', 'gato', 'cabra',
           'corço', 'cão', 'cão-guaxinim', 'gamo', 'esquilo', 'alce', 'equídeo', 'gineta', 'glutão', 
           'ouriço', 'lagomorfo', 'lobo', 'lontra', 'lince', 'mangusto', 'marmota', 'micromamífero', 'muflão', 
           'ovelha', 'mustelídeo', 'ave', 'urso', 'porco-espinho', 'nutria', 'rato-almiscarado', 
           'guaxinim', 'raposa', 'raposa-do-ártico', 'rena', 'javali', 'vaca']
}

txt_birdclasses = {
    'fr': ['anseriforme', 'autreoiseau', 'columbiforme', 'corvide', 'galliforme', 'passereau', 'piciforme', 'rapace'],
    'en': ['anseriform', 'otherbird', 'columbiform', 'corvid', 'galliform', 'passerine', 'piciform', 'raptor'],
    'it': ['anseriforme', 'altrouccello', 'columbiforme', 'corvide', 'galliforme', 'passeriforme', 'piciforme', 'rapace'],
    'de': ['Gänsevögel', 'Anderervögel', 'Taubenvögel', 'Rabenvögel', 'Hühnervögel', 'Sperlingsvögel', 'Spechtvögel', 'Greifvögel'],
    'es': ['anseriforme', 'otroave', 'columbiforme', 'córvido', 'galliforme', 'paseriforme', 'piciforme', 'rapaz'],
    'no': ['andefugler', 'andre fugler', 'duefugler', 'kråkefugler', 'hønsefugler', 'spurvefugler', 'spettefugler', 'rovfugler'],
    'se': ['andfåglar', 'andra fåglar', 'duvfåglar', 'kråkfåglar', 'hönsfåglar', 'tättingar', 'hackspettartade fåglar', 'rovfåglar'],
    'pt': ['anseriforme', 'outraave', 'columbiforme', 'corvídeo', 'galiforme', 'passeriforme', 'piciforme', 'rapinante']
}

####################################################################################
### CLASSIFIER
####################################################################################
class Classifier:
    def __init__(self, device=None):
        self.model = Model(device)
        self.model.loadWeights(DFVIT_WEIGHTS)
        self.transforms = transforms.Compose([
            transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))])

    def predictOnBatch(self, batchtensor, withsoftmax=True):
        return self.model.predict(batchtensor, withsoftmax)

    # croppedimage loaded by PIL
    def preprocessImage(self, croppedimage):
        preprocessimage = self.transforms(croppedimage)
        return preprocessimage.unsqueeze(dim=0)
    
class ClassifierWithBirds(Classifier):
    def __init__(self, device=None):
        Classifier.__init__(self, device)
        self.bird_model = BirdModel(device)
        self.bird_model.loadWeights(DFBIRD_WEIGHTS)
    
    def predictOnBatchBird(self, batchnumpy, withsoftmax=True):
        return self.bird_model.predict(tensor(batchnumpy), withsoftmax)

class BirdModel(nn.Module):
    def __init__(self, device=None):
        """
        Constructor of bird model classifier
        """
        super().__init__()
        self.bird_model = nn.Sequential(nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.GELU(),
                                        nn.Dropout(p=0.), nn.Linear(2048, 8))
        self.nbclasses = len(txt_birdclasses['fr'])
        self.device = device

    def predict(self, embeddings, withsoftmax=True):
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            x = embeddings.to(self.device)
            if withsoftmax:
                bird_predictions = self.bird_model.forward(x).softmax(dim=1)
            else:
                bird_predictions = self.bird_model.forward(x)
        return bird_predictions.cpu().numpy()

    def loadWeights(self, path):
        """
        :param path: path of .pt save of model
        """
        if path[-3:] != ".pt":
            path += ".pt"
        try:
            params = torch.load(path, map_location=self.device)
            # checkpoints may be stored in fp16 to save disk space; compute stays fp32
            params = {k: v.float() if torch.is_floating_point(v) else v for k, v in params.items()}
            self.load_state_dict(params)
        except Exception as e:
            print("Can't load bird checkpoint model because :\n\n " + str(e), file=sys.stderr)
            raise e

####################################################################################
### MODEL
####################################################################################

class Model(nn.Module):
    def __init__(self, device=None):
        """
        Constructor of model classifier
        """
        super().__init__()
         # Important: the model is trained with global_pool = 'token' (i.e. using the class token for final classification). 
         # This is not the default behavior with timm's dinov3 and thus should be specified when using timm.create_backbone:
        self.base_model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(txt_animalclasses['fr']),
                                            global_pool="token")
        print(f"Using {BACKBONE} for classification")
        self.backbone = BACKBONE
        self.nbclasses = len(txt_animalclasses['fr'])
        self.device = device

    def forward(self, input):
        x = self.base_model(input)
        return x

    def predict(self, data, withsoftmax=True):
        """
        Predict on test DataLoader
        :param test_loader: test dataloader: torch.utils.data.DataLoader
        :return: numpy array of predictions without soft max
        """
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            x = data.to(self.device)
            embeddings = self.base_model.forward_features(x)
            if withsoftmax:
                predictions = self.base_model.forward_head(embeddings).softmax(dim=1)
            else:
                predictions = self.base_model.forward_head(embeddings)
            embeddings = embeddings[:, 0, :]  # class token
        return predictions.cpu().numpy(), embeddings.cpu().numpy()

    def loadWeights(self, path):
        """
        :param path: path of .pt save of model
        """
        if path[-3:] != ".pt":
            path += ".pt"
        try:
            params = torch.load(path, map_location=self.device, weights_only=False)
            args = params['args']
            if self.nbclasses != args['num_classes']:
                raise Exception("You load a model ({}) that does not have the same number of class"
                                "({})".format(args['num_classes'], self.nbclasses))
            self.backbone = args['backbone']
            self.nbclasses = args['num_classes']
            # checkpoints may be stored in fp16 to save disk space; compute stays fp32
            state_dict = {k: v.float() if torch.is_floating_point(v) else v for k, v in params['state_dict'].items()}
            self.load_state_dict(state_dict)
        except Exception as e:
            print("Can't load checkpoint model because :\n\n " + str(e), file=sys.stderr)
            raise e
