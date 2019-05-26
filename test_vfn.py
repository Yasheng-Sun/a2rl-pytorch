import tensorflow as tf
import argparse
import vfn_network as nw
import numpy as np
import skimage.io as io
import skimage.transform as transform

global_dtype = tf.float32
global_dtype_np = np.float32
batch_size = 1

def evaluate_aesthetics_score(images):
    scores = np.zeros(shape=(len(images),))
    for i in range(len(images)):
        img = images[i].astype(np.float32)/255
        img_resize = transform.resize(img, (227, 227))-0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        scores[i] = sess.run([score_func], feed_dict={image_placeholder: img_resize})[0]
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", help="Embedding dimension before mapping to one-dimensional score", type=int, default = 1000)
    parser.add_argument("--initial_parameters", help="Path to initial parameter file", type=str, default="alexnet.npy")
    parser.add_argument("--ranking_loss", help="Type of ranking loss", type=str, choices=['ranknet', 'svm'], default='svm')
    parser.add_argument("--snapshot", help="Name of the checkpoint files", type=str, default='./snapshots/model-spp-max')
    parser.add_argument("--spp", help="Whether to use spatial pyramid pooling in the last layer or not", type=bool, default=True)
    parser.add_argument("--pooling", help="Which pooling function to use", type=str, choices=['max', 'avg'], default='max')

    args = parser.parse_args()

    embedding_dim = args.embedding_dim
    ranking_loss = args.ranking_loss
    snapshot = args.snapshot
    net_data = np.load(args.initial_parameters, encoding='latin1').item()
    image_placeholder = tf.placeholder(dtype=global_dtype, shape=[batch_size,227,227,3])
    var_dict = nw.get_variable_dict(net_data)
    SPP = args.spp
    pooling = args.pooling

    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim, SPP=SPP, pooling=pooling)
        score_func = nw.score(feature_vec)
    
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session(config=tf.ConfigProto())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, snapshot)

    images = [
        io.imread('./test_images/3846_cropped.jpg')[:,:,:3],   # remember to replace with the filename of your test image
        io.imread('./test_images/3846.jpg')[:,:,:3],
    ]

    # scores = evaluate_aesthetics_score(images)
    img_resize = transform.resize(images[0], (227, 227)) - 0.5
    img_resize = np.expand_dims(img_resize, axis=0)
    feature_vec = sess.run([feature_vec], feed_dict={image_placeholder: img_resize})[0]
    # print(scores)
    print(feature_vec.shape)