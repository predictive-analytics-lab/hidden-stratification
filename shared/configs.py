from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List


class CelebaAttributes(Enum):
    """CelebA attributes."""

    _5_o_Clock_Shadow = auto()
    Arched_Eyebrows = auto()
    Attractive = auto()
    Bags_Under_Eyes = auto()
    Bald = auto()
    Bangs = auto()
    Big_Lips = auto()
    Big_Nose = auto()
    Black_Hair = auto()
    Blond_Hair = auto()
    Blurry = auto()
    Brown_Hair = auto()
    Bushy_Eyebrows = auto()
    Chubby = auto()
    Double_Chin = auto()
    Eyeglasses = auto()
    Goatee = auto()
    Gray_Hair = auto()
    Heavy_Makeup = auto()
    High_Cheekbones = auto()
    Male = auto()
    Mouth_Slightly_Open = auto()
    Mustache = auto()
    Narrow_Eyes = auto()
    No_Beard = auto()
    Oval_Face = auto()
    Pale_Skin = auto()
    Pointy_Nose = auto()
    Receding_Hairline = auto()
    Rosy_Cheeks = auto()
    Sideburns = auto()
    Smiling = auto()
    Straight_Hair = auto()
    Wavy_Hair = auto()
    Wearing_Earrings = auto()
    Wearing_Hat = auto()
    Wearing_Lipstick = auto()
    Wearing_Necklace = auto()
    Wearing_Necktie = auto()
    Young = auto()


class IsicAttrs(Enum):
    """Attributes available for the ISIC dataset."""

    histo = auto()
    malignant = auto()
    patch = auto()


class AdultDatasetSplit(Enum):
    """Adult dataset splits."""

    Sex = auto()
    Race = auto()
    Race_Binary = auto()
    Race_Sex = auto()
    Custom = auto()
    Nationality = auto()
    Education = auto()


class QuantizationLevel(Enum):
    """Quantization level."""

    three = 3
    five = 5
    eight = 8


@dataclass
class DatasetConfig:
    """General data set settings."""

    # _target_: str = "shared.configs.DatasetConfig"
    log_name: str  # don't rely on this to check which dataset is loaded

    data_pcnt: float = 1.0  # data pcnt should be a real value > 0, and up to 1
    context_pcnt: float = 0.4
    test_pcnt: float = 0.2
    root: str = ""
    transductive: bool = False  # whether to include the test data in the pool of unlabelled data


@dataclass
class AdultConfig(DatasetConfig):
    """Settings specific to the Adult dataset."""

    _target_: str = "shared.configs.AdultConfig"
    log_name: str = "adult"

    # Adult data set feature settings
    drop_native: bool = True
    adult_split: AdultDatasetSplit = AdultDatasetSplit.Sex
    drop_discrete: bool = False
    adult_balanced_test: bool = True
    balance_all_quadrants: bool = True


@dataclass
class ImageDatasetConfig(DatasetConfig):
    """Settings specific to image datasets."""

    quant_level: QuantizationLevel = QuantizationLevel.eight  # number of bits that encode color
    input_noise: bool = False  # add uniform noise to the input


@dataclass
class CmnistConfig(ImageDatasetConfig):
    """Settings specific to the cMNIST dataset."""

    _target_: str = "shared.configs.CmnistConfig"
    log_name: str = "cmnist"

    # Colored MNIST settings
    scale: float = 0.0
    greyscale: bool = False
    background: bool = False
    black: bool = True
    binarize: bool = True
    rotate_data: bool = False
    shift_data: bool = False
    color_correlation: float = 1.0
    padding: int = 2  # by how many pixels to pad the cmnist images by
    filter_map_labels: Dict[str, int] = field(default_factory=dict)
    colors: List[int] = field(default_factory=list)


@dataclass
class CelebaConfig(ImageDatasetConfig):
    """Settings specific to the CelebA dataset."""

    _target_: str = "shared.configs.CelebaConfig"
    log_name: str = "celeba"

    # CelebA settings
    celeba_sens_attr: CelebaAttributes = CelebaAttributes.Male
    celeba_target_attr: CelebaAttributes = CelebaAttributes.Smiling


@dataclass
class IsicConfig(ImageDatasetConfig):
    """Settings specific to the ISIC dataset."""

    _target_: str = "shared.configs.IsicConfig"
    log_name: str = "isic"

    # ISIC settings
    isic_sens_attr: IsicAttrs = IsicAttrs.histo
    isic_target_attr: IsicAttrs = IsicAttrs.malignant


@dataclass
class BiasConfig:

    # _target_: str = "shared.configs.BiasConfig"

    # Dataset manipulation
    missing_s: List[int] = field(default_factory=list)
    mixing_factor: float = 0  # How much of context should be mixed into training?
    adult_biased_train: bool = True  # if True, make the training set biased, based on mixing factor
    # the subsample flags work like this: you give it a class id and a fraction in the form of a
    # float. the class id is given by class_id = y * s_count + s, so for binary s and y, the
    # correspondance is like this:
    # 0: y=0/s=0, 1: y=0/s=1, 2: y=1/s=0, 3: y=1/s=1
    subsample_context: Dict[str, float] = field(default_factory=dict)
    subsample_train: Dict[str, float] = field(default_factory=dict)

    log_dataset: str = ""


@dataclass
class BaseConfig:
    """Minimum needed config to do data loading."""

    data: DatasetConfig
    bias: BiasConfig
