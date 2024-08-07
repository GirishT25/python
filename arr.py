arr =[1, 1, 1, 0, 0, 0,0]
for i in range(len(arr) - 1):
    if arr[i]==0 and arr[i+1]==1:
        print("Invaid format")
    elif arr[i]==1 and arr[i]==0:
        pass
    elif arr[i]==1 and arr[i]==1:
        pass
low = 0 
high =len(arr) - 1
mid = (low + high)//2
while low<=high:
    mid =(low +high)//2
    if arr[mid] == 1:
        low = mid +1
    else :
        high = mid - 1 
print(f"Number of zeros ",len(arr)-low)