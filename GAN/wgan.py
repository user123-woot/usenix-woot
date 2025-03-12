"""
Here we put classes regarding wagn_gp architecture 
feature selection among 6 advesarial features (igonre src2dst/dst2src_min_ps)
- Use the correlation matrix of benign samples to pick feature pairs that remain realistic after perturbation.
---
Positive changes
- layer architecture like mirror G(256/128/64) D(64/128/256/1) 
- add monitoring
- add combiner and reordering to give proper feature order in batch 
critic
- last layer is linear
- according to paper not batch normalization used
- leakyrelu
- input data normalized using z score
- add kernel_initializer=initializers.HeNormal() to critic too all layers
generator
- use batch normalization + use_bias=False
- last layer sofplus==> to make it smoother and not generating 0 always like relu or negative like leakyrelu
- add "He" weight initialization + last layer

---
Keep track of changes during babysitting 
penatly: 10
critic lr: 0.00004 
critic epoch: 2 
generator output is sofplus ==> to make it smoother and not generating 0 always like relu or negative like leakyrelu
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input, concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import yaml, os, time

def gradient_penalty(critic, real_samples, fake_samples):
    #Computes gradient penalty 
    alpha = tf.random.uniform([real_samples.shape[0], 1], 0.0, 1.0) #  generates random values (coefficients) from a uniform distribution between 0 and 1.
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples # creates interpolated samples by linearly combining real and fake samples using the coefficients alpha. 
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        validity = critic(interpolates)
    gradients = tape.gradient(validity, [interpolates])[0]
    grad_norm = tf.norm(gradients, ord=2, axis=1)  # L2 norm
    gradient_penalty = tf.reduce_mean((grad_norm - 1.0) ** 2)
    #print(f"gradient_penalty ==> {gradient_penalty}")
    return gradient_penalty
class Critic: # discriminator
    # according to the paper no batsch normalization is recommended for critic
    def __init__(self, input_size, lr, lambda_gp=10): # lambda_gp used in the original paper 10
        self.input_size = input_size
        self.lambda_gp = lambda_gp  # Weight for gradient penalty
        self.optimizer = Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.critic = self.build_critic()
        print(self.critic.summary())
    def build_critic(self): # mirror architecture to generator: G(256/128/64) D(64/128/256/1) 
        model = Sequential(name="Critic")
        model.add(Dense(64, input_shape=(self.input_size,),kernel_initializer=initializers.HeNormal()))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128, kernel_initializer=initializers.HeNormal()))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256, kernel_initializer=initializers.HeNormal()))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, kernel_initializer=initializers.HeNormal()))  # No activation function as it should be linear for gradient penalty calculation
        return model

    def compute_loss(self, real_samples, fake_samples):
        real_loss = K.mean(self.critic(real_samples))
        fake_loss = K.mean(self.critic(fake_samples))
        gp = gradient_penalty(self.critic, real_samples, fake_samples)
        return fake_loss - real_loss + self.lambda_gp * gp #difference between the fake loss and the real loss, plus the gradient penalty

class Generator:
    def __init__(self, lr, latent_dim=100, mutable_size=1, immutable_size=53):
        self.latent_dim = latent_dim
        self.mutable_size = mutable_size
        self.immutable_size = immutable_size
        self.optimizer = Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.generator = self.build_generator()

    def build_generator(self):
        model = Sequential(name="Generator")
        model.add(Dense(256, input_shape=(self.latent_dim,), use_bias=False,
                        kernel_initializer=initializers.HeNormal())) # when using BN we should set bias to false
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128,use_bias=False,
                        kernel_initializer=initializers.HeNormal()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(64, use_bias=False,
                        kernel_initializer=initializers.HeNormal()))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # using lenear activation function for output layer to produce real number wihtout constraint to activation function
        #model.add(Dense(self.mutable_size, activation="softplus")) 
        model.add(Dense(self.mutable_size, activation="softplus", kernel_initializer=initializers.HeNormal())) 
      
        #model.add(Dense(self.mutable_size, activation=(LeakyReLU(alpha=0.2))))  #Using relu to generrate pertube value >=0, however it can be tanh where many use but need define a transformation  
        return model
    """   
class WGAN_GP:
    def __init__(self, generator, critic, lr, lambda_gp=10):
    
        # build WGAN-GP model with generator, critic, optimizer, and gradient penalty .
        
        self.generator = generator
        self.critic = critic
        self.lambda_gp = lambda_gp
        self.optimizer = Adam(learning_rate=lr, beta_1=0.5, beta_2=0.9)
        self.wgan_gp = self.build_wgan_gp()
        print(self.wgan_gp.summary())
    def build_wgan_gp(self):
      
        # Build the combined WGAN-GP model.
    
        noise_input = Input(shape=(self.generator.latent_dim,), name="latent_noise")  # Latent space
        immutable_features = Input(shape=(self.generator.immutable_size,), name="immutable_features")  # Fixed features
        
        # Generate adversarial features
        adversarial_features = self.generator.generator(noise_input)

        # Ensure proper concatenation of immutable and generated features
        fake_sample = concatenate([immutable_features, adversarial_features], axis=-1)
        
        # Ensure critic input shape is correct
        validity = self.critic.critic(fake_sample)
        
        return Model(inputs=[noise_input, immutable_features], outputs=validity, name="WGAN_GP")
   
def get_real_samples(batch_size): # sample from real data (benign and malicioius) with size of batch 
    df = pd.read_csv("/home/mehrdad/PycharmProjects/GAN-Framework/dataset/red-team2/deployment_ds/red-team2_merge.csv")
    df_benign = df[df["label"]==0].sample(int(batch_size/2))
    df_malicious = df[df["label"]==1].sample(int(batch_size/2))
    real_data = pd.concat([df_benign,df_malicious], axis=0).sample(frac=1) # shuffel
    real_data = real_data.loc[:, df.columns != 'label']
    return real_data
    
    """
class train_wgan_gp:
    def __init__(self,df,generator, critic,adversarial_feature , result_folder, epochs=1000, batch_size=64, n_critic=5):
        self.df = df
        self.generator = generator
        self.critic =critic
        self.adversarial_feature = adversarial_feature
        self.result_folder = result_folder
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_critic=n_critic
        self.critic_losses = []
        self.generator_losses = []
        self.used_indices = set()

        
    def get_real_samples(self):
        # Separate benign and malicious data
        df_benign = self.df[self.df["label"] == 0]
        df_malicious = self.df[self.df["label"] == 1]
        
        # Check if there is enough data left for sampling
        if len(df_benign) <= self.batch_size / 2 or len(df_malicious) <= self.batch_size / 2:
            raise ValueError("Not enough data for one of the labels to complete the batch sampling.")

        # Ensure we do not reuse previously selected data
        df_benign_sampled = df_benign.loc[~df_benign.index.isin(self.used_indices)].sample(int(self.batch_size/2))
        df_malicious_sampled = df_malicious.loc[~df_malicious.index.isin(self.used_indices)].sample(int(self.batch_size/2))
        
        # Combine the benign and malicious samples
        real_data = pd.concat([df_benign_sampled, df_malicious_sampled], axis=0).sample(frac=1)  # Shuffle
        
        # Update the used_indices to include the current sampled rows
        self.used_indices.update(real_data.index)
        # Drop the 'label' column before returning
        real_data = real_data.loc[:, real_data.columns != 'label']
        
        return real_data
    def get_malicious_samples(self):
      
        # Filter malicious samples
        df_malicious = self.df[self.df["label"] == 1]
        
        # Ensure there are enough malicious samples left for the requested batch size
        if len(df_malicious) <= self.batch_size:
            raise ValueError("Not enough malicious data left to complete the sampling.")
        
        # Ensure we do not reuse previously selected malicious data
        df_malicious_sampled = df_malicious.loc[~df_malicious.index.isin(self.used_indices)].sample(self.batch_size)
        
        # Update the used indices with the newly sampled rows
        self.used_indices.update(df_malicious_sampled.index)

        # Drop the 'label' column before returning
        fake_data = df_malicious_sampled.loc[:, df_malicious_sampled.columns != 'label']
        
        return fake_data

    def combiner(self,immutable_part,perturbed_adversarial_feature ):
        # the concat pf purturbe and immutable part should be based on correct original column order 
        reference_df = pd.read_csv("/home/mehrdad/PycharmProjects/GAN-Framework/dataset/red-team2/deployment_ds/red-team2_merge.csv")
        reference_df.drop('label', axis=1, inplace=True)  
        perturbed_df = pd.DataFrame(perturbed_adversarial_feature, columns=self.adversarial_feature)
        
        immutable_part = immutable_part.reset_index(drop=True)
        perturbed_df = perturbed_df.reset_index(drop=True)
        combined_df = pd.concat([immutable_part, perturbed_df], axis=1)

        # Ensure the columns are in the original dataset order
        combined_df = combined_df[reference_df.columns]

        return combined_df
    def reorder_and_combine(self,immutable_features, perturbed_adversarial_feature,  reference_columns):
        feature_dict = {}  # Store tensors by column name

        # Ensure all tensors are float32 (or a common dtype)
        immutable_features = tf.cast(immutable_features, tf.float32)
        perturbed_adversarial_feature = tf.cast(perturbed_adversarial_feature, tf.float32)

        # Split immutable features into separate tensors
        immutable_columns = [col for col in reference_columns if col not in self.adversarial_feature]
        split_immutable = tf.split(immutable_features, num_or_size_splits=len(immutable_columns), axis=1)

        # Store immutable features in the dictionary
        for i, col in enumerate(immutable_columns):
            feature_dict[col] = split_immutable[i]

        # Split and store perturbed adversarial features
        perturbed_split = tf.split(perturbed_adversarial_feature, num_or_size_splits=len(self.adversarial_feature), axis=1)
        for i, col in enumerate(self.adversarial_feature):
            feature_dict[col] = perturbed_split[i]

        # Reorder based on reference_columns and concatenate
        ordered_tensors = [feature_dict[col] for col in reference_columns]
        combined_tensor = tf.concat(ordered_tensors, axis=1)

        return combined_tensor  



    def log_losses(self,epoch, critic_loss, gen_loss):
        self.critic_losses.append(critic_loss)
        self.generator_losses.append(gen_loss)

        # Save loss values
        #np.save("critic_losses.npy", np.array(critic_losses))
        #np.save("generator_losses.npy", np.array(generator_losses))

        # Plot losses every 10 epochs
        if epoch % 30 == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(self.critic_losses, label="Critic Loss")
            plt.plot(self.generator_losses, label="Generator Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("GAN Training Progress")
            plt.savefig(f"{self.result_folder}training_progress_epoch_{epoch}.png")  # Save plot
            #plt.show()

    def normalize_critic_input(self,data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-6  # Avoid division by zero
        return (data - mean) / std


    def train_wgan_gp(self):
        for epoch in range(self.epochs):
            try:
                for _ in range (self.n_critic):  # Train Critic more often
                    try:
                        real_samples = self.get_real_samples() # both benign and malicious ==>balanced

                        noise = np.random.normal(0, 1, (self.batch_size, self.generator.latent_dim))
                        perturbed_adversarial_feature = self.generator.generator.predict(noise) 

                        fake_samples= self.get_malicious_samples()  
                        immutable_part = fake_samples.loc[:, ~fake_samples.columns.isin(self.adversarial_feature)]  
                        
                        fake_samples_final = self.combiner(immutable_part,perturbed_adversarial_feature ) 
                        # np.concatenate([immutable_part, perturbed_adversarial_feature], axis=1)  # Shape: (batch_size, 54)

                        # Train Critic
                        with tf.GradientTape() as tape:
                            loss = self.critic.compute_loss(self.normalize_critic_input(real_samples), self.normalize_critic_input(fake_samples_final))
                        grads = tape.gradient(loss, self.critic.critic.trainable_variables)
                        self.critic.optimizer.apply_gradients(zip(grads, self.critic.critic.trainable_variables))
                    except ValueError as e:
                    # Exception raised if there is not enough data for either benign or malicious samples
                        print(f"Error: {e}. Stopping sampling.")
                        break
                #for _ in range(2):
                # Train Generator
                self.critic.critic.trainable = False  

                fake_samples = self.get_malicious_samples()  # Shape: (g_batch_size, 53)

                noise = np.random.normal(0, 1, (self.batch_size, self.generator.latent_dim))
                immutable_features = fake_samples.loc[:, ~fake_samples.columns.isin(self.adversarial_feature)]  # Shape: (batch_size, 53)
                
                with tf.GradientTape() as tape:
                    # 🔹 Corrected fake sample input for generator training
                    perturbed_feature = self.generator.generator(noise)  # Shape: (batch_size, 1)
                    fake_samples_gen = self.reorder_and_combine( # combine immutable and pertube and return in tensof format 
                        immutable_features=immutable_features,
                        perturbed_adversarial_feature=perturbed_feature,
                            reference_columns=list(self.df.columns[self.df.columns != 'label'])) # to ensure the order 
                    gen_loss = -K.mean(self.critic.critic(self.normalize_critic_input(fake_samples_gen))) 

                    g_grads = tape.gradient(gen_loss, self.generator.generator.trainable_variables)
                    self.generator.optimizer.apply_gradients(zip(g_grads, self.generator.generator.trainable_variables))
                    
                self.critic.critic.trainable = True  
                self.log_losses(epoch=epoch,critic_loss=loss.numpy(), gen_loss=gen_loss.numpy()) #***********
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Critic Loss: {loss.numpy()}, Generator Loss: {gen_loss.numpy()}")
                    print(f"perturbed_feature : {perturbed_feature}")
                
                if epoch >= 40 and epoch % 10 ==0: # from epoch 50 appened every 10 
                    noise = np.random.normal(0, 1, (1000, self.generator.latent_dim))
                    perturbed_feature = self.generator.generator(noise)
                    if perturbed_feature.shape[1] == 1:
                        pertube_df = pd.DataFrame(perturbed_feature, columns=self.adversarial_feature)  # 1-column case
                    else:
                        pertube_df = pd.DataFrame(perturbed_feature, columns=self.adversarial_feature)
                    csv_file_path = self.result_folder + f"{self.adversarial_feature}_epoch_{epoch}_criticLoss-{loss.numpy()}_genLoss-{gen_loss.numpy()}.csv"
                    if os.path.exists(csv_file_path):
                        # Append the DataFrame to the existing CSV file without writing headers
                        pertube_df.to_csv(csv_file_path, mode='a', header=False, index=False)
                    else:
                        # Write the DataFrame to the CSV file with headers
                        pertube_df.to_csv(csv_file_path, mode='w', header=True, index=False)
            except ValueError as e:
                # Exception raised if there is not enough data for either benign or malicious samples
                print(f"Error: {e}. Stopping sampling.")
                break
def main():

    with open("/home/mehrdad/PycharmProjects/C2_communication/GAN/gan_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    df = pd.read_csv("/home/mehrdad/PycharmProjects/C2_communication/dataset/combined-benign-malcious/benign_deepred-autoc2-2-3-2025-001.csv").loc[:,config["features"]]
    currenc_time= time.strftime("%d-%m-%Y-%H-%M-%S")
    os.mkdir(f"/home/mehrdad/PycharmProjects/C2_communication/GAN/results/{currenc_time}")

    for adv_feature in config["repeated_adversarial_features"]:
        os.mkdir(f"/home/mehrdad/PycharmProjects/C2_communication/GAN/results/{currenc_time}/{adv_feature}")
        result_folder = f"/home/mehrdad/PycharmProjects/C2_communication/GAN/results/{currenc_time}/{adv_feature}/"
        latent_dim = 100
        mutable_size = len(adv_feature)
        immutable_size = df.shape[1]-mutable_size - 1 #1 is for label
        print("=================================================")
        print(f"latent_dim : {latent_dim} mutable_size: {mutable_size} immutable_size:{immutable_size}")
        print("=================================================")

        generator = Generator(lr=0.001, latent_dim=latent_dim, mutable_size=mutable_size, immutable_size=immutable_size)
        critic = Critic(input_size=mutable_size + immutable_size, lr=0.00004,lambda_gp= 10 )

        #wgan_gp = WGAN_GP(generator, critic, lr=0.001)

        #----------------Start training
        train = train_wgan_gp(df=df,generator=generator, critic=critic, adversarial_feature=adv_feature, 
                    epochs=200, batch_size=64, n_critic=2, result_folder=result_folder)
        train.train_wgan_gp()

if __name__ == "__main__":
    main()