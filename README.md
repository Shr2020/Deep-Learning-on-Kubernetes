## Objective
Develop Container and Kubernetes artefacts perform DL training and DL inference hosting in
IBM Kubernetes cluster.

### Steps Environment Setup:
1. Create the VPC on IBM Cloud
2. Create the k8 cluster
3. On the local VM, install all the necessary packages to run IBM cloud CLI, kubectl and
minikube (to test locally). I have used the vagrant VM (that was used for the Kubernetes
lab) to install all necessary packages mentioned in the slides.

### Steps for Development:
#### DOCKER RUN:
1. Developed the deep learning model for MNIST data classification.
2. The files containing the code for training and inference of the model are train.py and
inference.py.
3. The files containing code for the frontend are front.html and backend.html
4. Two Dockerfiles are used to create two containers: one for train and one for inference.
The train container trains the model and saves the model weights. These weights are
then used by the inference container and the prediction/classification is done.
5. The container images are pushed to the docker hub. (Repository: sjdocker3409/k8_dl)
6. Command For running the containers locally (without the Kubernetes), the working
directory of the container was mapped to a local directory while docker was run for
training. The same was done while the docker run for inference along with mapping the
port of the container to the localhost port.
7. Then access the server at `http://localhost:39000/`
8. Commands used:
 1. `sudo docker run -it -v /home:/mnist mnist_train:latest`
 2. `sudo docker run -it -v /home:/mnist -p 39000:9000 mnist_inference:latest`

#### DEPLOYMENT ON MINIKUBE
1. Once the containers were working as required. Tested the program locally on minikube.
2. Created for 4 yaml files :
deployment.yaml,
train.yaml,
service.yaml,
kustomization.yaml.
3. The train container was run using train.yaml. Create a Pod (kind:Pod) for training
purposes. In the yaml file mounted a local directory to the directory where the
model weights will save.
4. The inference container was run using deployment.yaml. Created a deployment
for inference (kind: Deployment) and mounted the same local directory as in the
above steps.
5. The service was created using service.yaml (kind: Service, type: LoadBalancer).
This created a service and mapped the port of deployment to an external port so
the URL can be accessed from outside.
6. Also created a kustomization.yaml.
7. To get the external IP address of service. Use the command : `minikube
service <servicename> --url`
8. Can access the URL : `http://<external-ip>:<node-port>/`


#### DEPLOYMENT ON IBM CLOUD:
1. Log in to IBM cloud and select your resource group.
2. Run the command: `ibmcloud ks cluster config -c <cluster-id>`
3. Now we can use kubectl on IBM cluster.
4. Create a new yaml file pvc.yaml. This one is to use the Persistent Volume Claim on IBM
Cloud.
5. Create the train.yaml, inference.yaml, service.yaml and customization.yaml as before. In
train.yaml and inference.yaml mount pvc, to be used as the shared volume between the
containers.
6. Run the following commands in order:
 1. `cd <directory which contains all yaml files>`
 2. `kubectl apply -f pvc.yaml`. Wait till the pvc is up and ready.
 3. `kubectl apply -f train.yaml` Wait till the pod has completed training.
 4. `kubectl apply -f deployment.yaml`
 5. `kubectl apply -f service.yaml`
 6. `kubectl get service`. Get the external IP address. The port will the the “port” specified in the service.yaml.
