from tensorflow import keras
from keras import layers
from keras import regularizers

conv_depth_1 = 15
conv_depth_2 = 25 
conv_depth_3 = 50 
conv_depth_4 = 50
fc_nodes = 512

drop_prob_1 = 0.25
drop_prob_2 = 0.5
drop_prob_3 = 0.25 
drop_prob_4 = 0.5
l2_penalty = 1e-3

#number of depth images to keep, only a certain number contains the base gangli
zmin = 20
zmax = 30

#portion of image which contains the interested part of brain scan
xmin = 33
xmax = 162
ymin = 73
ymax = 202

deltaX = xmax-xmin
deltaY = ymax-ymin
deltaZ = zmax-zmin


def conv_pool_drop_block(
    input,
    num_filters,
    drop_rate,
    l2_penalty,
    kernel_size=3,
    padding='same',
    activation='relu',
    pool_size=2,
    kernel_regularizer=regularizers.L2(l2_penalty),
):

    x = layers.Conv3D(
        num_filters, 
        kernel_size=kernel_size,
        padding=padding,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        data_format='channels_last',
        )(input)

    x = layers.MaxPooling3D(pool_size=pool_size)(x)
    x = layers.Dropout(drop_rate)(x)

    return x


def fc_block(
    input,
    num_layers,
    fc_nodes,
    drop_rate,
    l2_penalty,
    kernel_regularizer=regularizers.L2(l2_penalty),
    activation='relu',
):
    x = layers.Flatten()(input)

    for fc_layers in range(num_layers):
        x = layers.Dense(
            fc_nodes,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
            )(x)
        x = layers.Dropout(drop_rate)(x)
    
    return x
    

def build_model(
    img_size,
    num_classes,
):
    input = keras.Input(shape=(img_size, img_size, deltaZ, 1))

    x = conv_pool_drop_block(input, num_filters=conv_depth_1, drop_rate=drop_prob_1, l2_penalty=l2_penalty)
    x = conv_pool_drop_block(x, num_filters=conv_depth_2, drop_rate=drop_prob_2, l2_penalty=l2_penalty)
    x = conv_pool_drop_block(x, num_filters=conv_depth_3, drop_rate=drop_prob_3, l2_penalty=l2_penalty)
    x = layers.Conv3D(
        conv_depth_4, 
        (3,3,3), 
        padding='same', 
        activation='relu', 
        data_format='channels_last', 
        kernel_regularizer=regularizers.L2(l2_penalty))(x)
    x = layers.Dropout(drop_prob_4)(x)

    x = fc_block(x, num_layers=2, fc_nodes=fc_nodes, drop_rate=drop_prob_4, l2_penalty=l2_penalty)
    
    output = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs=input, outputs=output)