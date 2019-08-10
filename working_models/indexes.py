def euclidean( a, b = None):
  if b is None:
      b = a
  return np.sum(a*b)

def NDWI(image, limit):
    NDW_idx = np.divide((image[2,:,:]-image[7,:,:]),(image[2,:,:]+image[7,:,:]))
    IsWater=(np.logical_not(NDW_idx > limit)).astype(int)
    return( IsWater )

def NDVI(image, limit):
    NDV_idx = np.divide((image[7,:,:]-image[3,:,:]),(image[3,:,:]+image[7,:,:])).astype('float')
    IsVegi=(np.logical_not(NDV_idx > limit)).astype(int)
    return(IsVegi)
