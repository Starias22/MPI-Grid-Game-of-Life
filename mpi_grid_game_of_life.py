import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI


def distribute_points(N, p):
    """
    Distribute points among a given number of elements.
    """
    
    eq = N // p  # Calculate equal distribution among elements
    remainder = N % p  # Calculate remainder after equal distribution
    arr = np.full(p, eq,dtype='i')  # Initialize array with equal distribution

    for i in range(remainder,0,-1):
        arr[i] += 1

    return arr



def find_best_dimensions(nx, ny, num_procs):
    """This function finds the best decompositon basing on the aspect ratio of the domain"""
    aspect_ratio = nx / ny
    best_ratio = float('inf')
    best_dims = None
    
    for i in range(1, num_procs + 1):
        if num_procs % i == 0:
            j = num_procs // i
            if abs(aspect_ratio - i / j) < best_ratio:
                best_ratio = abs(aspect_ratio - i / j)
                best_dims = [i, j]
    
    return np.array(best_dims,dtype='i')



def get_neighbors(cart):

    
    low, high = cart.Shift(direction=1, disp=1)
    
    left, right = cart.Shift(direction=0, disp=1)
    neighbors=[high,left,low,right]
    
    return neighbors

def get_next_state(augmented_grid,cell):
    """This function takes as arguments an augmented grid and a cell, which is a list of row and colomn
    indexes in the real grid. Then, it returns the next state of the grid"""
    
    # Get the row and column indexes of the cell
    row_index,col_index=cell

    
   

    top_neighbors=augmented_grid[row_index-1,col_index-1:col_index+2] # States of the top cells
    bottom_neighbors=augmented_grid[row_index+1,col_index-1:col_index+2] # States of the bottom cells
    beside_neighbors=augmented_grid[row_index, col_index-1],augmented_grid[row_index, col_index+1] # States of the cells beside
    

    
    
    num=np.sum(top_neighbors)+np.sum(bottom_neighbors)+ np.sum(beside_neighbors)  #Count alive neighbors
     
    state=None
    if num==0 or num==1 or num>=4:
        state=0 # The cell die
    elif num==3:
        state=1 # The cell becomes populated
    else:# ie if num==2 
        state=augmented_grid[row_index,col_index] # The cell keeps it's state
    #print('Top neighbors  {} Bottom neighbors {} Beside neighbors {} num={} and state={}'.format(top_neighbors,bottom_neighbors,beside_neighbors,num,state))
    
    return state

def get_next_grid(augmented_grid):
    
    """This function takes as argument an augmented grid and then computes and returns the 
    next grid, which contains the state of the grid in the next iteration"""
    # Get the shape of the augmented grid
    nrows,ncols=augmented_grid.shape
    
    next_grid =augmented_grid.copy()
    
    # Get the next state of each cell in the grid
    for i in range(1,nrows-1):
        for j in range(1,ncols-1):
            next_grid[i,j]= get_next_state(augmented_grid,[i,j])
            
    
    return next_grid


def exchange(augmented_grid):
    
    # Cell states exchange between north and south
    sendbuf = augmented_grid[1,1:-1].copy()
    recvbuf1=-1 *np.ones(nx_cells,dtype=np.int8)
    recvbuf2=-1 *np.ones(nx_cells,dtype=np.int8)

    
    
    dest=source=north
    COMM.Sendrecv(
            sendbuf=[ sendbuf ,1,type_line1],
            dest=dest,
            recvbuf=[ recvbuf1 ,1,type_line1]
        ,source=source
            )

    sendbuf = augmented_grid[-2,1:-1].copy()
    dest=source=south
    COMM.Sendrecv(
            sendbuf=[ sendbuf ,1,type_line1],
            dest=dest,
            recvbuf=[ recvbuf2 ,1,type_line1],
            source=source
            )

    if np.all(recvbuf1 == -1):
        recvbuf1=np.zeros_like(recvbuf2)

    recvbuf1 = recvbuf1[ recvbuf1!=-1]
    
    augmented_grid[0,1:-1]=recvbuf1.copy()


    if np.all(recvbuf2 == -1):
        recvbuf2=np.zeros_like(recvbuf2)


    augmented_grid[-1,1:-1]=recvbuf2.copy()


     # Cell states exchange between est and west

    sendbuf = augmented_grid[1:-1:,1].copy()
    recvbuf1=-1 *np.ones(ny_cells,dtype=np.int8)
    recvbuf2=-1 *np.ones(ny_cells,dtype=np.int8)

    dest=source=est
    COMM.Sendrecv(
            sendbuf=[ sendbuf ,1,type_line2],
            dest=dest,
            recvbuf=[ recvbuf1 ,1,type_line2]
        ,source=source
            )



    sendbuf = augmented_grid[1:-1:,-2].copy()
    dest=source=west
    COMM.Sendrecv(
            sendbuf=[ sendbuf ,1,type_line2],
            dest=dest,
            recvbuf=[ recvbuf2 ,1,type_line2],
            source=source
            )


    if np.all(recvbuf1 == -1):
        recvbuf1=np.zeros_like(recvbuf2)

    recvbuf1 = recvbuf1[ recvbuf1!=-1]

    augmented_grid[1:-1:,0]=recvbuf1.copy()


    if np.all(recvbuf2 == -1):
        recvbuf2=np.zeros_like(recvbuf2)
        
    augmented_grid[1:-1,-1]=recvbuf2.copy()

    
def group_procs_by_coord(cart,SIZE,axis):
    
    procs_grouped_by_coord = {}
    for rank in range(SIZE): # For each process
        coords = cart.Get_coords(rank)  # Get the coordinates of the process 
        key = coords[axis]  
        if key not in procs_grouped_by_coord:
            procs_grouped_by_coord[key] = []
        procs_grouped_by_coord[key].append(rank)
    return procs_grouped_by_coord



# Define neighbors indexes(constants)
N = 0
E = 1
S = 2
W = 3



def life_game(width: int, height: int, delay: float=0.1, nit: int=50,reproduce: bool=False)-> None:
    """
    Simulates Conway's Game of Life.

    Parameters:
        width (int): The width of the game grid.
        height (int): The height of the game grid.
        delay (float, optional): The time delay between iterations in seconds. Defaults to 0.1.
        nit (int, optional): The number of iterations. Defaults to 50.
        reproduce (bool, optional): The initial states of the cells grid are initialized at random.
          If reproduce==True the initial values of the cells doesnt differ from one execution of the Game to another.

    Returns:
        None
    """

    
    # Initialize the communication world
    global COMM
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    global RANK
    RANK = COMM.Get_rank()
    nprocs=np.zeros(2,dtype='i')
    
    if RANK==0:
        print('Game of life running with {} cells along x ,{} cells along y, delay={} and {} iterations'.format(width,height,delay,nit))
        # The dimensions of the grid
        

        # The number of processes along each axis(x and y)
        nprocs=find_best_dimensions(width,height, SIZE)
        #print('{} processes along x and {} processes along y'.format(nprocs[0],nprocs[1])) 

    COMM.Bcast([nprocs, 2,MPI.INT],root=0) # Broadcast nprocs to all the processes

           

    cols_nums=np.empty(nprocs[0],dtype='i')
    rows_nums=np.empty(nprocs[1],dtype='i')

    if RANK==0:
        nprocs_x,nprocs_y=nprocs


        cols_nums=distribute_points(width,nprocs_x)

        rows_nums=distribute_points(height,nprocs_y)

    # Send the numbers of cells along x and y axes to each process
    COMM.Bcast([cols_nums, nprocs[0],MPI.INT],root=0)
    COMM.Bcast([rows_nums, nprocs[1],MPI.INT],root=0)





    # Create cartesian 2D cart
    global cart
    cart=COMM.Create_cart(dims = nprocs)

    # Group processes by x-axis
    procs_by_coord=group_procs_by_coord(cart,SIZE,axis=0)
    
    global nx_cells,ny_cells

    # Get the number of cells along x
    i=0
    for x_coord in procs_by_coord:
        procs=procs_by_coord[x_coord]
        if RANK in procs:
            nx_cells=cols_nums[i]
            break
        i+=1

    # Group processes by y-axis
    procs_by_coord=group_procs_by_coord(cart,SIZE,axis=1)

    # Get the number of cells along y
    i=0
    for y_coord in procs_by_coord:
        procs=procs_by_coord[y_coord]
        if RANK in procs:
            ny_cells=rows_nums[i]
            break
        i+=1    



    #print('Process {} with nx_cells={} and ny_cells={}'.format(RANK,nx_cells,ny_cells))

    # Number of cells along x and y including (the 2) ghost cells
    countx=nx_cells+2
    county=ny_cells+2

    # Create contigous types for data exchanges
    global type_line1,type_line2
    type_line1= MPI.CHAR.Create_contiguous(nx_cells)
    type_line1.Commit()
    
    type_line2= MPI.CHAR.Create_contiguous(ny_cells)
    type_line2.Commit()

    # Get neighbors
    neighbors= get_neighbors(cart)
    global north,s
    global north,est,south,west
    north,est,south,west=neighbors



    # Create the grid including ghost cells
    augmented_grid=-1*np.ones((county,countx),dtype=np.int8)
    # Initialize cell states at random
    if reproduce:
        # Fix the numpy random generator
        np.random.seed(0)
    augmented_grid[1:-1,1:-1]=np.random.randint(2,size=(ny_cells,nx_cells),dtype=np.int8)

    it=0

    if RANK==0:
    
        plt.xlim(0,width)
        plt.ylim(0,height)

    while it<=nit:


        local_grid=augmented_grid[1:-1,1:-1]
        #print('Process {} with local grid={}'.format(RANK,local_grid))
        recvbuf= COMM.gather(local_grid, root=0)#The process 0 gathers the local grids


        if RANK==0:

            # Group processes ranks by y-coordinate
            procs_group=group_procs_by_coord(cart,SIZE,axis=1)


            grid=[]

            # Concatenate local grids along y-axis
            for y in procs_group:
                arr=np.concatenate([recvbuf[i] for i in procs_group[y]],axis=1)
                grid.append(arr)


            # Concatenate local grid along x-axis
            grid=np.concatenate(grid,axis=0)
            #print('At it={}, global grid=\n{}:'.format(it,grid))

            # Show the curent states of cells
            
            plt.imshow(grid)
            #plt.imshow(grid,cmap='binary')
            plt.title('Game of life iteration {}/{}'.format(it,nit))

            plt.show(block=False)
            plt.pause(delay)
        
        # Exchange cell states with neighbors
        exchange(augmented_grid)

        # Compute the next states of cells
        augmented_grid=get_next_grid(augmented_grid)

        # Move to the next iteration
        it+=1

    # Stay on the last state of the grid
    if RANK==0:
        
        plt.imshow(grid)
        plt.title('Game of life last iteration: Press Q to quit')
        plt.show()
        
    # Free the datatypes    
    type_line1.Free()
    type_line2.Free()


    
#life_game(width=40,height=30,delay=1,nit=10) 
#life_game(width=16,height=5,delay=1,nit=10)

#life_game(width=16,height=5,reproduce=True)  #delay is 0.1 nit=50 and reproduice=False by default 

life_game(width=250,height=100,nit=50,delay=0.01)  #delay is 0.1 by default
