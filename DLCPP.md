# DLCPP

## Step 1. **Initializing Phase**
Given training samples X and desired number of hidden layers
- labeled data at the node
  - amount of all content being being requested at node i at t slot time **(request (t)i )**
  - number of content *o* being requested at node i t slot time **(content_request(o,t)i)**
  - the entropy of content type at node i during the t slot time **(request_entropy(t)i)**
  - the number of content type at node i during the t slot time **(content_sum(t)i)**

- labeled common data
  - the amount of all contents being requested by all nodes at t slot time **(request_allnode(t))**
  - the amount of content *o* being requested by all nodes during t slot time **(content_request_allnode(o,t))**
  - post time for t slot **(timestamp)**

