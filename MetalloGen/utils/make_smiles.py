import os
import numpy as np

from MetalloGen import chem
from MetalloGen.utils import frag

visit=[] 
visited_V=[]
def dfs(BO,vertex):
    """Perform (recursive) Depth First Search.
    BO - Bond order (or adjacency) matrix
    vertex - Index of the atom to start DFS.

    returns: 'explicitly' nothing, but it yields the global variable 'visited_V'. 
    Finally, visited_V contains atom indices of one of the molecules in an intermediate.
    """
    global visit,visited_V
    visited_V.append(vertex)
    visit[vertex]=1
    for i in range(len(BO)):
        if BO[vertex][i]!=0 and visit[i]==0: 
            dfs(BO,i)
    return

def start_new_explore():
    """After a single round of DFS finishes, it finds a new starting point to continue DFS of the NEW molecule.
    It uses the global variable 'visit'.
    returns: the smallest index of unvisited atoms, -1 if all atoms in an intermediate were visited
    """
    global visit
    for i in range(len(visit)):
        if visit[i]==0: 
            return i
    return -1

def reduceBO(BO,Elemlist):
    """ It reduces an atomic connectivity matrix which contains plural molecules into the matrix corresponding to each single molecule.
    BO - Bond order (or adjacency) matrix
    Elemlist - List of atom elements in an intermediate state (e.g. ['C','O','H','H'])

    returns:
        reducedlist - lists of original indices of atoms in each single molecule (e.g. [[0,3], [1,2]])
        reducedBOs -  lists of matrices of each single molecule (e.g. [ [[0,1],[1,0]], [[0,1],[1,0]] ] )
        reducedELs -  lists of atom elements of each single molecule (e.g. [['C','O'],['H','H']])
    """
    global visit,visited_V
    reducedlist=[]
    visit=np.zeros((len(BO)))
    while True:
        visited_V=[]
        vertex=start_new_explore()
        if vertex==-1: 
            break
        dfs(BO,vertex)
        reducedlist.append(visited_V)
    reducedBOs=[]
    reducedELs=[]
    for i in range(len(reducedlist)):
        reducedBO=np.zeros((len(reducedlist[i]),len(reducedlist[i])))
        reducedEL=[]
        for j in range(len(reducedlist[i])):
            for k in range(len(reducedlist[i])):
                reducedBO[j][k]=BO[reducedlist[i][j]][reducedlist[i][k]]
            reducedEL.append(Elemlist[reducedlist[i][j]])
        reducedBOs.append(reducedBO)
        reducedELs.append(reducedEL)
    return reducedlist, reducedBOs, reducedELs



def dfs_SMILES(BO,atom_list,vertex,bondlist): 
    """It performs a nonrecursive depth-first search for finding main chain, branch, ring openers in a molecule.
    This routine is necessary for generating a SMILES string from an atomic connectivity matrix.
    (Please refer to Bull. Korean Chem. Soc. 2015, 36, 1759.)

    BO - Bond order (or adjacency) matrix
    atom_list - list of atom class instances
    vertex - Index of the atom to start DFS.
    bondlist - List of bonds included in the minimum spanning tree (MST) of a given molecule (please refer to def kruskal).
               Each bond is given as a tuple of two atom indices.
    
    returns:
         main_chain - list of indices of atoms corresponding to the main chain 
         branches - dictionary. Each key represents the index of 'branch root', 
                    and contains the list of indices of atoms corresponding to each branch.
         Ring_openers - List of bonds corresponding to 'ring openers'. 
                        It is obtained by omitting bonds included in the MST from the original atomic connectivity. 
    """
    
    #Ring_openers=map(lambda x:  tuple(x), bondlist[:])
    Ring_openers=bondlist[:]
    visit=np.zeros((len(BO)))
    main_chain=[] 
    branches=dict()
    stack=[] 
    parents=[]
    stack.insert(0,vertex)
    parents.insert(0,-1) #no parent
    Ring_openers.append((-1,vertex))
    ismainchain=True

    while stack!=[]:
        current=stack.pop(0)
        current_parent=parents.pop(0)
        if visit[current]==0:
            visit[current]=1
            if ismainchain: 
                main_chain.append(current)
            else: 
                branches[branch_root][len(branches[branch_root])-1].append(current) #append to the last branch

            Ring_openers.remove((min(current,current_parent),max(current,current_parent)))
            stack_insertcount=0
            for i in range(len(BO)-1,-1,-1):
                if BO[current][i]!=0 and visit[i]==0 and (i not in stack) and ( (str.upper(atom_list[i].element)!='H') or (str.upper(atom_list[i].element)=='H' and atom_list[i].is_divalent_hydrogen==True)):
                    stack.insert(0,i)
                    parents.insert(0,current)
                    stack_insertcount+=1
            if stack_insertcount==0 and parents!=[]: #branch
                ismainchain=False
                branch_root=parents[0]
                try: #for multiple branches. add new branch
                    DoesBranchExist=branches[branch_root]
                    branches[branch_root].append([])
                except KeyError: #new branch
                    branches[branch_root]=[[]]
    #if len(bondlist)!=0: print(bondlist) ########

    return main_chain,branches,Ring_openers

def Different_set(atom1,atom2,sets):
    """ Routine used in the Kruskal algorithm.

    atom1, atom2 - Indices of two atoms of a bond.
    sets - A collection of lists, each of which corresponds to separated trees generated during the Kruskal algorithm.

    returns:
        True   if at least one atom does not belong to any trees (i.e. Adding the atom1-atom2 bond does not form any rings!)
        False  otherwise
    """
    for i in range(len(sets)):
        if sets[i].count(atom1)==1 and sets[i].count(atom2)==1:
            return False
    return True

def Union_set(atom1,atom2,sets):
    """ Routine used in the Kruskal algorithm.

    atom1, atom2 - Indices of two atoms of a bond.
    sets - A collection of lists, each of which corresponds to separated trees generated during the Kruskal algorithm.

    returns:
        sets - Updated collection of lists, each of which corresponds to separated trees generated during the Kruskal algorithm.
    """
    atom1_setindex=-1
    atom2_setindex=-1
    for i in range(len(sets)):
        if sets[i].count(atom1)==1: atom1_setindex=i
        if sets[i].count(atom2)==1: atom2_setindex=i

    if atom1_setindex==-1 and atom2_setindex==-1: #Both of them are not in set yet
        sets.append([atom1,atom2])
    elif atom1_setindex==-1 and atom2_setindex!=-1: #one of them not in set
        sets[atom2_setindex].append(atom1)
    elif atom1_setindex!=-1 and atom2_setindex==-1: #one of them not in set
        sets[atom1_setindex].append(atom2)
    else: #both are in different set
        set1=sets[atom1_setindex]
        set2=sets[atom2_setindex]
        sets.append(set1+set2)
        sets.remove(set1)
        sets.remove(set2)
    return sets

def kruskal(atom_list,BO,edges,ExcludeTerminalHydrogens=True):
    """ Kruskal algorithm.

    atom_list - List of atom class instances
    BO - Bond order matrix 
    edges - List of edges (i.e. chemical bonds) of a molecule. Each bond is given as a tuple of two atom indices
    ExcludeTerminalHydrogens - Whether to exclude terminal hydrogens or not during applying the Kruskal algorithm.

    returns: 
        Ring_opener - List of chemical bonds corresponding to 'ring openers' (i.e. bonds excluded through Kruskal algorithm)
        tree - List of chemical bonds included in the minimum spanning tree (MST)
    """ 
    weights=[]
    new_edges=[]
    for i in range(len(edges)):
        atom1=edges[i][0]
        atom2=edges[i][1]
        if not ExcludeTerminalHydrogens: 
            case1=case2=case3=True
        elif atom_list!='':
            case1=(str.upper(atom_list[atom1].element)!='H' and str.upper(atom_list[atom2].element)!='H')
            case2=(str.upper(atom_list[atom1].element)=='H' and atom_list[atom1].is_divalent_hydrogen)
            case3=(str.upper(atom_list[atom2].element)=='H' and atom_list[atom2].is_divalent_hydrogen)
        else: case1=case2=case3=True
        if case1 or case2 or case3:
            weight=1.0/BO[atom1][atom2]
            weights.append(weight)
            new_edges.append(edges[i])

    for i in range(len(new_edges)):
        for j in range(i+1,len(new_edges)):
            if weights[i]>weights[j]:
                temp=weights[j]
                weights[j]=weights[i]
                weights[i]=temp
                temp=new_edges[j]
                new_edges[j]=new_edges[i]
                new_edges[i]=temp

    Ring_opener=new_edges[:]
    tree=[]
    vertex_sets=[]
    for i in range(len(weights)):
        atom1=new_edges[i][0]
        atom2=new_edges[i][1]
        if Different_set(atom1,atom2,vertex_sets):
            tree.append(new_edges[i])
            Ring_opener.remove(new_edges[i])
            vertex_sets=Union_set(atom1,atom2,vertex_sets)

    return Ring_opener,tree

def get_SMILES_for_one_atom(index,neighbor_bo,atom_list,BO,charge,rings,RSorEZ=''):
    """ Generate a segment of the SMILES string. 
    The whole SMILES string is generated by concatenating all the segments generated in this routine.

    index - index of an atom whose SMILES segment will be generated
    neighbor_bo - order of the bond formed with the neighboring atom. ('' for single, '=' for double, '#' for triple bonds)
    atom_list - list of atom class instances 
    BO - Bond order matrix 
    charge - formal charge of the current target atom
    rings - identifiers for rings, if the current target atom is relevant to specification of rings.
            they are generated in def make_ring_identifiers
    RSorEZ - 'RS' if an atom corresponds to the chiral center. 'EZ' if an atom corresponds to cis-trans center. '' otherwise.

    returns: 
        oneatom_smiles - a segment of the SMILES string
        numH - 'H#' where '#' is the number of hydrogen atoms binded to the current target atom. 
    """
    organics=['B','C','N','O','P','S','F','Cl','Br']
    if abs(charge)==0 or abs(charge)==1: 
        chg=''
    else: 
        chg=str(abs(charge))

    if charge>0: 
        chg='+'+chg
    elif charge<0: 
        chg='-'+chg
    else:
        chg=''
    element=str.lower(atom_list[index].element)
    element=str.upper(element[0])+element[1:] #Only first letter uppercase

    numH=count_Explicit_hydrogens(index,atom_list,BO)
    Explicit_hydrogen='H'+str(numH)
    if Explicit_hydrogen=='H0': 
        Explicit_hydrogen=''
    elif Explicit_hydrogen=='H1':
        Explicit_hydrogen='H'

    try:
        is_organic=organics.index(element)
        if charge==0:
            if RSorEZ=='RS':
                if numH==0: 
                    Explicit_hydrogen=''
                    numH='H0'
                elif numH==1: 
                    Explicit_hydrogen='H'
                    numH='H1'  #two hydrogens? not possible for stereogenic center!!
                oneatom_smiles=neighbor_bo+'['+element+'RS'+Explicit_hydrogen+']'
            else: oneatom_smiles=neighbor_bo+'['+element+Explicit_hydrogen+']'+RSorEZ #All Atom with bracket notation
            #else: oneatom_smiles=neighbor_bo+element+RSorEZ
            if element=='C': #special case for carbene 
                if list(BO[index]).count(1)==2 and list(BO[index]).count(2)==0 : #two single bonds with no multiple bond
                    if numH==2: 
                        oneatom_smiles='[CH2]'
                        numH='H2'
                    elif numH==1: 
                        oneatom_smiles='[CH]'
                        numH='H1'
                    else: 
                        oneatom_smiles='[C]'
            elif element=='B' and numH==0: 
                oneatom_smiles=neighbor_bo+'['+element+']' 
            elif (element=='F' or element=='Cl' or element=='Br') and numH==0: 
                oneatom_smiles='['+element+']'
            elif len(atom_list)==1: 
                oneatom_smiles='['+element+']'
        else:
            oneatom_smiles=neighbor_bo+'['+element+Explicit_hydrogen+chg+']'
            #numH='H'+str(numH)
    except ValueError:
        Explicit_hydrogen='H'+str(numH)
        if Explicit_hydrogen=='H0':
            Explicit_hydrogen=''
        elif Explicit_hydrogen=='H1':
            Explicit_hydrogen='H'
        oneatom_smiles=neighbor_bo+'['+element+Explicit_hydrogen+chg+']'
        #numH='H'+str(numH)


    if type(numH)==str and numH[0]!='H': 
        numH='H'+str(numH)  #NOW ALL HYDROGENS ARE EXPLICIT!!! (150527)
    if type(numH)==int: numH='H'+str(numH)  #NOW ALL HYDROGENS ARE EXPLICIT!!! (150527)
    oneatom_smiles+=rings[index] #if there is a ring
    return oneatom_smiles, numH

def make_ring_identifiers(Num_atoms,Ring_openers):
    """ It generates 'identifiers' of rings of SMILES string.
    Num_atoms - Number of atoms in a molecule
    Ring_openers - 'Ring_openers' obtained by Kruskal algorithm

    returns:
        Rings - list of ring identifiers for each atom in a molecule. 
                e.g. ['','12','%10',...]  '' indicates that this atom is not relevant to specification of a ring.
    """
    Rings=[]
    Rings+=Num_atoms*['']
    for i in range(len(Ring_openers)):
        Ring_count=i+1
        A1=Ring_openers[i][0]
        A2=Ring_openers[i][1]
        if Ring_count>=10:
            Rings[A1]+='%'+str(Ring_count)
            Rings[A2]+='%'+str(Ring_count)
        else:
            Rings[A1]+=str(Ring_count)
            Rings[A2]+=str(Ring_count)
    return Rings

def count_Explicit_hydrogens(index,atom_list,BO):
    """ It counts the number of hydrogen atoms attached to the specific atom in a molecule.

    index - index of an atom whose number of hydrogens will be counted
    atom_list - list of atom class instances 
    BO - Bond order matrix 

    returns:
        num_H - number of hydrogens (integer)
    """
    num_H=0
    for i in range(len(BO)):
        if BO[index][i]!=0 and str.upper(atom_list[i].element)=='H': 
            #151127
            if int(np.sum(BO[i]))==1:
                num_H+=1 #ordinary case
            '''
            elif int(np.sum(BO[i]))==2: #divalent hydrogen?
                indices=[a for a, x in enumerate(list(BO[i])) if int(x)==1]

                #if indices[0]==index: num_H+=1
                if indices[1]==index: num_H+=1
            '''
    return num_H


def Detect_stereocenter(atom_list,bondlist,BO): #For carbons only
    """ It finds stereocenter atoms in a molecule.

    atom_list - list of atom class instances
    bondlist - list of bonds in a molecule. Each bond is given as a tuple of two atom indices.

    returns:
        EZcenters - list of cis-trans (or E-Z) center atoms (e.g. [[0,3],[5,8]])
        RScenters - list of chiral center atoms (e.g. [0,3,4]) 
    """
    RScenters=[]
    EZcenters=[]
    SN4=[]
    SN3=[]
    for i in range(len(atom_list)):
        if str.upper(atom_list[i].element)=='C':
            SN=np.count_nonzero(BO[i])
            if SN==4:
                SN4.append((i,np.nonzero(BO[i])[0]))
            elif SN==3:
                SN3.append((i,np.nonzero(BO[i])[0]))

    olefins=[]
    for i in range(len(SN3)):
        for j in range(i+1,len(SN3)):
            center1=SN3[i][0]
            center2=SN3[j][0]
            #if BO[center1][center2]!=0:
            if BO[center1][center2]==2:
                BFS_roots=[]
                for k in SN3[i][1]:
                    if k!=center2: 
                        BFS_roots.append(k)
                for k in SN3[j][1]:
                    if k!=center1: 
                        BFS_roots.append(k)
                olefins.append((center1,center2,BFS_roots))

    EZcenters=Detect_EZ(atom_list,bondlist,olefins)
    RScenters=Detect_RS(atom_list,bondlist,SN4)

    #print(SN4)
    #print(olefins)
    #print(EZcenters)
    #print(RScenters)
    return EZcenters,RScenters


def Detect_EZ(atom_list,bondlist,olefins):
    """ It finds cis-trans (or E-Z) center atoms in a molecule.

    atom_list - list of atom class instances
    bondlist - list of bonds in a molecule. Each bond is given as a tuple of two atom indices.
    olefins - list of information on the (R1)(R2)C1=C2(R3)(R4) bonds. 
              [(index of C1, index of C2, [indices of atoms in R1,R2,R3,R4 connected to C1 and C2]), (...), ...]

    returns:
        EZcenters - list of cis-trans (or E-Z) center atoms (e.g. [[0,3],[5,8]])
    """
    EZcenters=[]
    Adj=frag.makeAdjacency(len(atom_list),bondlist)
    Zlist=[x.get_atomic_number() for x in atom_list]

    for CCdouble in olefins:
        center1=CCdouble[0]
        center2=CCdouble[1]
        BFS_roots=CCdouble[2]

        newAdj=np.copy(Adj)
        newAdj[:,center1]=np.zeros(len(Adj))
        newAdj[center1]=np.zeros(len(Adj)) #replace to zeros
        newAdj[:,center2]=np.zeros(len(Adj))
        newAdj[center2]=np.zeros(len(Adj)) #replace to zeros

        reducedlist, reducedAdjs, reducedZlists=reduceBO(newAdj,Zlist)

        G1root=BFS_roots[0]
        G2root=BFS_roots[1]
        G3root=BFS_roots[2]
        G4root=BFS_roots[3]

        reducedlist.pop(reducedlist.index([center1])) #remove C=C atoms
        reducedlist.pop(reducedlist.index([center2]))

        for i in range(len(reducedlist)):
            if reducedlist[i].count(G1root)!=0: 
                G1rootindex=i
            if reducedlist[i].count(G2root)!=0: 
                G2rootindex=i
            if reducedlist[i].count(G3root)!=0: 
                G3rootindex=i
            if reducedlist[i].count(G4root)!=0: 
                G4rootindex=i

        cond1=(G1rootindex==G3rootindex)
        cond2=(G1rootindex==G4rootindex)
        cond3=(G2rootindex==G3rootindex)
        cond4=(G2rootindex==G4rootindex)

        if cond1 or cond2 or cond3 or cond4: 
            continue ##CCdouble bond in a ring. not EZ center. pass

        cond5=(G1rootindex==G2rootindex)
        cond6=(G3rootindex==G4rootindex) #G1-G2 connected, G3-G4 connected

        if cond5 and not cond6:
            part3=reducedlist[G3rootindex][:]
            part4=reducedlist[G4rootindex][:]
            segment=reducedlist[G1rootindex]
            visited=[0]*len(segment)
            visited[segment.index(G1root)]=1
            visited[segment.index(G2root)]=1
            parts=multibfs([[G1root],[G2root]],[[G1root],[G2root]],segment,visited,newAdj)
            part1=parts[0]
            part2=parts[1]
        elif cond6 and not cond5:
            part1=reducedlist[G1rootindex][:]
            part2=reducedlist[G2rootindex][:]
            segment=reducedlist[G3rootindex]
            visited=[0]*len(segment)
            visited[segment.index(G3root)]=1
            visited[segment.index(G4root)]=1
            parts=multibfs([[G3root],[G4root]],[[G3root],[G4root]],segment,visited,newAdj)
            part3=parts[0]
            part4=parts[1]
        elif cond5 and cond6:
            segment=reducedlist[G1rootindex]
            visited=[0]*len(segment)
            visited[segment.index(G1root)]=1
            visited[segment.index(G2root)]=1
            parts=multibfs([[G1root],[G2root]],[[G1root],[G2root]],segment,visited,newAdj)
            part1=parts[0]
            part2=parts[1]
            segment=reducedlist[G3rootindex]
            visited=[0]*len(segment)
            visited[segment.index(G3root)]=1
            visited[segment.index(G4root)]=1
            parts=multibfs([[G3root],[G4root]],[[G3root],[G4root]],segment,visited,newAdj)
            part3=parts[0]
            part4=parts[1]
        else:
            part1=reducedlist[G1rootindex][:]
            part2=reducedlist[G2rootindex][:]
            part3=reducedlist[G3rootindex][:]
            part4=reducedlist[G4rootindex][:]

        #Comparing Ceigs of (G1,G2), and those of (G3 G4). Different -> E or Z center
        part1.sort()
        part2.sort()
        part3.sort()
        part4.sort()
        part1Adj=newAdj[:,part1][part1,:]
        part1Zlist=[Zlist[x] for x in part1]
        part2Adj=newAdj[:,part2][part2,:]
        part2Zlist=[Zlist[x] for x in part2]
        part3Adj=newAdj[:,part3][part3,:]
        part3Zlist=[Zlist[x] for x in part3]
        part4Adj=newAdj[:,part4][part4,:]
        part4Zlist=[Zlist[x] for x in part4]

        C1=frag.getCoulombic(part1Zlist,part1Adj)
        C2=frag.getCoulombic(part2Zlist,part2Adj)
        C3=frag.getCoulombic(part3Zlist,part3Adj)
        C4=frag.getCoulombic(part4Zlist,part4Adj)

        if not frag.isSameMolecule(C1,C2) and not frag.isSameMolecule(C3,C4): 
            EZcenters.append([center1,center2])
    return EZcenters


def Detect_RS(atom_list,bondlist,SN4):
    """ It finds chiral center atoms in a molecule.

    atom_list - list of atom class instances
    bondlist - list of bonds in a molecule. Each bond is given as a tuple of two atom indices.
    SN4 -  list of information on the tetravalent carbon ( C(R1)(R2)(R3)(R4) )
           [(index of C, [indices of atoms in R1,R2,R3,R4 connected to C]), (...), ...]

    returns:
        RScenters - list of chiral center atoms (e.g. [0,3,4])
    """
    RScenters=[]
    Adj=frag.makeAdjacency(len(atom_list),bondlist)
    #Zlist=[am.getZ(x.element) for x in atom_list]
    Zlist=[x.get_atomic_number() for x in atom_list]

    for fourarms in SN4:
        center=fourarms[0]
        roots=fourarms[1]
        newAdj=np.copy(Adj)
        newAdj[:,center]=np.zeros(len(Adj))
        newAdj[center]=np.zeros(len(Adj))
        reducedlist, reducedAdjs, reducedZlists=reduceBO(newAdj,Zlist)

        reducedlist.pop(reducedlist.index([center])) #remove C=C atoms

        Fourgroups_Adjs=[]
        Fourgroups_Zlists=[]

        for i in range(len(reducedlist)):
            bfs_roots=[]
            for root in roots:
                if reducedlist[i].count(root)!=0:
                    bfs_roots.append(root)
            if len(bfs_roots)==1: #no need to bfs
                onegroup=reducedlist[i][:]
                onegroup.sort()
                onegroup_Adj=newAdj[:,onegroup][onegroup,:]
                onegroup_Zlist=[Zlist[x] for x in onegroup]
                Fourgroups_Adjs.append(onegroup_Adj)
                Fourgroups_Zlists.append(onegroup_Zlist)
            else:
                segment=reducedlist[i]
                visited=[0]*len(segment)
                for bfs_root in bfs_roots: 
                    visited[segment.index(bfs_root)]=1
                queues=[[x] for x in bfs_roots]
                parts=queues[:]
                parts=multibfs(queues,parts,segment,visited,newAdj)

                for part in parts:
                    onegroup=part[:]
                    onegroup.sort()
                    onegroup_Adj=newAdj[:,onegroup][onegroup,:]
                    onegroup_Zlist=[Zlist[x] for x in onegroup]
                    Fourgroups_Adjs.append(onegroup_Adj)
                    Fourgroups_Zlists.append(onegroup_Zlist)

        Fourgroups_Ceigs=[]
        for i in range(4): 
            Fourgroups_Ceigs.append(frag.getCoulombic(Fourgroups_Zlists[i],Fourgroups_Adjs[i]))

        Chiral=True
        for i in range(4):
            for j in range(i+1,4):
                if frag.isSameMolecule(Fourgroups_Ceigs[i],Fourgroups_Ceigs[j]): 
                    Chiral=False
                    break
            if not Chiral: 
                break
        if Chiral: 
            RScenters.append(center)

    return RScenters


def multibfs(queues,parts,segment,visited,Adj):
    """ A stepwise breadth-first search (BFS) algorithm performed simultaneously from multiple starting points 

    queues - Multiple number of queues used in the BFS algorithm. Initially it consists of lists of starting atoms. 
            (e.g. [[0],[3]])
    parts -  Multiple number of lists. The indices of atoms visited by the BFS will be appended to each list.
             Initially it consists of lists of starting atoms. (e.g. [[0],[3]])
    segment - Indices of all atoms which will be subject to BFS
    visited - list of zeros and ones. 1 if an atom corresponds to the starting atom of BFS, 0 otherwise. (e.g. [1,0,0,1,0,0,0,0,0,0,0])
    Adj - Adjacency matrix of a molecule.

    returns: 
        parts - lists of indices of atoms searched through the BFS algorithm.
        (e.g. [[0,1,2,4,6,7],[3,5,8,9,10]])
    """
    while True:
        newqueues=[]
        for queue in queues:
            newqueue=[]
            for atom1 in queue:
                for atom2 in range(len(Adj)):
                    if Adj[atom1][atom2]!=0 and segment.count(atom2)!=0 and visited[segment.index(atom2)]==0:
                        newqueue.append(atom2)
            newqueues.append(newqueue)

        if len(newqueues)==2:
            queue1= list(set(newqueues[0]) - set(newqueues[1]))
            queue2= list(set(newqueues[1]) - set(newqueues[0]))
            for atom in list(set(newqueues[0]) | set(newqueues[1])):
                visited[segment.index(atom)]=1
            queues=[queue1,queue2]
        elif len(newqueues)==3:
            queue1= list(set(newqueues[0]) - set(newqueues[1]) - set(newqueues[2]))
            queue2= list(set(newqueues[1]) - set(newqueues[2]) - set(newqueues[0]))
            queue3= list(set(newqueues[2]) - set(newqueues[0]) - set(newqueues[1]))
            for atom in list(set(newqueues[0]) | set(newqueues[1]) | set(newqueues[2])):
                visited[segment.index(atom)]=1
            queues=[queue1,queue2,queue3]
        elif len(newqueues)==4:
            queue1= list(set(newqueues[0]) - set(newqueues[1]) - set(newqueues[2]) - set(newqueues[3]))
            queue2= list(set(newqueues[1]) - set(newqueues[2]) - set(newqueues[3]) - set(newqueues[0]))
            queue3= list(set(newqueues[2]) - set(newqueues[3]) - set(newqueues[0]) - set(newqueues[1]))
            queue4= list(set(newqueues[3]) - set(newqueues[0]) - set(newqueues[1]) - set(newqueues[2]))
            for atom in list(set(newqueues[0]) | set(newqueues[1]) | set(newqueues[2]) | set(newqueues[3])):
                visited[segment.index(atom)]=1
            queues=[queue1,queue2,queue3,queue4]

        for i in range(len(parts)):
            parts[i]+=queues[i]

        if sum(visited)==len(visited) or sum([len(x) for x in queues])==0:
            return parts

def GetSMILES(atom_list_original,BO,bondlist,fc,Find_Stereocenter):
    """ It converts the bond order matrix into the SMILES string.

    atom_list_original -  list of atom class instances. 3D coordinates are not necessary. Atom types should be provided at least 
    BO - Bond order matrix of a molecule
    bondlist - list of bonds in a molecule. Each bond is given as a tuple of two atom indices.
    fc - list of formal charges of each atom in a molecule. (e.g. [0,0,0,1,0,0,-1,0]
    Find_Stereocenter - if 'Y', generate SMILES strings with considering chiral centers, cis-trans (E-Z) centers. Otherwise, do not consider stereochemistry 

    returns:
        SMILES - a SMILES string.
        atom_labels - It contains information on correspondence of atom indices of the bond order matrix with those of SMILES string
                      (e.g. [3,'H','H',0,'H'] indicates that the atom 3 in BO matrix corresponds to the atom 0 in SMILES,
                      and atoms 1, 2, 4 are indices of hydrogen atoms in SMILES)
                      This is used for matching atom numberings between BO matrix and the 3D geometry generated by pybel.
                      (Please refer to def.permuteBO and def.UFFrelax)
        NumRings - Number of rings in a molecule. 
    """
    #find nearest nonhydrogen
    for i in range(len(atom_list_original)):
        if str.upper(atom_list_original[i].element)!='H': 
            start=i
            break
    SMILES=''
    nonH_bondlist=[]
    for i in range(len(bondlist)):
        atom1=str.upper(atom_list_original[bondlist[i][0]].element)
        atom2=str.upper(atom_list_original[bondlist[i][1]].element)
        case1=(atom1=='H' and np.count_nonzero(BO[bondlist[i][0]])>=2) #Divalent hydrogens included - (151130)
        case2=(atom2=='H' and np.count_nonzero(BO[bondlist[i][1]])>=2) #Divalent hydrogens included - (151130)
        case3=( atom1!='H' and atom2!='H' )
        if case1: 
            atom_list_original[bondlist[i][0]].set_is_divalent_hydrogen(True) 
        if case2: 
            atom_list_original[bondlist[i][1]].set_is_divalent_hydrogen(True) 
        if case1 or case2 or case3:
            nonH_bondlist.append(bondlist[i]) #We don't need hydrogens when searching for rings

    #bondlist=nonH_bondlist[:]

    try:
        Ring_openers,tree=kruskal(atom_list_original,BO,nonH_bondlist) #switch all ring openers to single bonds by kruskal algorithm which finds minimum spanning tree
        #print(Ring_openers)
        MST_BO=np.zeros((len(atom_list_original),len(atom_list_original)))
        for j in range(len(tree)):
            atom1=tree[j][0]
            atom2=tree[j][1]
            MST_BO[atom1][atom2]=MST_BO[atom2][atom1]=BO[atom1][atom2]
        main_chain,branches,dummy=dfs_SMILES(MST_BO,atom_list_original,start,tree)
        Rings=make_ring_identifiers(len(atom_list_original),Ring_openers)
        if Find_Stereocenter=='Y':
            EZcenters,RScenters=Detect_stereocenter(atom_list_original,bondlist,BO)
            SMILES,atom_labels=encoding_to_SMILES(main_chain,branches,Rings,atom_list_original,BO,fc,SMILES,start,[],[],EZcenters,RScenters)
        else:
            SMILES,atom_labels=encoding_to_SMILES(main_chain,branches,Rings,atom_list_original,BO,fc,SMILES,start,[],[])
    except UnboundLocalError: #In this case, all atoms are hydrogens. Just begin from the first atom.
        main_chain,branches,Ring_openers=dfs_SMILES(BO,atom_list_original,0,nonH_bondlist)
        Rings=make_ring_identifiers(len(atom_list_original),Ring_openers)
        SMILES,atom_labels=encoding_to_SMILES(main_chain,branches,Rings,atom_list_original,BO,fc,SMILES,0,[],[])

    SMILES=SMILES[1:len(SMILES)-1] #main chain does not need '(' ')' 
    NumRings=len(Ring_openers)
    return SMILES,atom_labels,NumRings

def make_stereoSMILES(SMILESlist,num_stereocenter):
    """ It generates SMILES strings of all possible stereoisomers of the given SMILES.

    SMILESlist - a list of SMILES string which contains 'RS' or 'EZ' mark
                (e.g. ['N[CRSH](C)C(=O)O'])
    num_stereocenter - Number of stereo centers

    returns: 
        SMILESlist - a list of (2^num_stereocenter) SMILES strings
                (e.g. ['N[C@H](C)C(=O)O', 'N[C@@H](C)C(=O)O'])
    """
    if num_stereocenter==0: 
        return SMILESlist
    while True:
        newSMILESlist=[]
        for i in range(len(SMILESlist)):
            SMILES=SMILESlist[i]
            for j in range(len(SMILES)):
                part=SMILES[j:j+3]
                if len(part)>=2 and part[:2]=='RS':
                    newSMILES1=SMILES[:j]+'@'+SMILES[j+2:]
                    newSMILES2=SMILES[:j]+'@@'+SMILES[j+2:]
                    break
                elif len(part)==3 and part[:2]=='EZ':
                    EZindex=int(part[2])
                    for k in range(j+1,len(SMILES)):
                        part2=SMILES[k:k+3]
                        if len(part2)==3 and part2[:2]=='EZ' and int(part2[2])==EZindex:
                            newSMILES1=SMILES[:j]+'/'+SMILES[j+3:k]+'/'+SMILES[k+3:]
                            newSMILES2=SMILES[:j]+'/'+SMILES[j+3:k]+'\\'+SMILES[k+3:]
                            break
                    break
            newSMILESlist.append(newSMILES1)
            newSMILESlist.append(newSMILES2)
        SMILESlist=newSMILESlist
        #print(len(SMILESlist))
        if len(SMILESlist)==2**num_stereocenter:
            return SMILESlist


def encoding_to_SMILES(segment,branches,rings,atom_list,BO,fc,SMILES,branch_root,atom_labels,EZcenters=[],RScenters=[]):
    """ Using the information on a molecule, it encodes a molecule into SMILES string. 
    It is a recursive function.

    segment - list of atom indices correspoding to the main chain or any branch of a molecule
    branches - branches in a molecule. Generated by def.dfs_SMILES
    rings - rings in a molecule. Generated by def.make_ring_identifiers 
    atom_list - list of atom class instances. Atom types should be provided at least
    BO - Bond order matrix 
    fc - list of formal charges of each atom in a molecule. (e.g. [0,0,0,1,0,0,-1,0]
    SMILES - 'current' SMILES string. This is a recursive function and SMILES string will be accumulating recursively.
    branch_root - the starting atom of a branch 
    atom_labels - Please see below
    EZcenters - list of cis-trans (or E-Z) center atoms (e.g. [[0,3],[5,8]])
    RScenters - list of chiral center atoms (e.g. [0,3,4])

    returns:
        SMILES - SMILES string
        atom_labels - It contains information on correspondence of atom indices of the bond order matrix with those of SMILES string
                      (e.g. [3,'H','H',0,'H'] indicates that the atom 3 in BO matrix corresponds to the atom 0 in SMILES,
                      and atoms 1, 2, 4 are indices of hydrogen atoms in SMILES)
                      This is used for matching atom numberings between BO matrix and the 3D geometry generated by pybel.
                      (Please refer to def.permuteBO and def.UFFrelax)
    """
    if segment==[]:
        return SMILES
    SMILES+='('
    for i in range(len(segment)):
        RSorEZ=''
        now=segment[i]
        if i==0: 
            prev=branch_root
        else: 
            prev=segment[i-1]

        bo=BO[now][prev]
        if bo==3: 
            neighbor_bo='#'
        elif bo==2: 
            neighbor_bo='='
        else: 
            neighbor_bo=''
        charge=fc[now]

        if RScenters.count(now)!=0: 
            RSorEZ='RS'
        for j in range(len(EZcenters)):
            if EZcenters[j].count(now)!=0: 
                RSorEZ='EZ'+str(j)

        if RScenters==[] and EZcenters==[]: 
            oneatom_smiles,numH=get_SMILES_for_one_atom(now,neighbor_bo,atom_list,BO,charge,rings)
        else: 
            oneatom_smiles,numH=get_SMILES_for_one_atom(now,neighbor_bo,atom_list,BO,charge,rings,RSorEZ)

        SMILES+=oneatom_smiles
        atom_labels.append(now)

        #print(numH)
        if type(numH)==str and numH[0]=='H':
            Num_Explicit_Hydrogens=int(numH[1:])
            for i in range(Num_Explicit_Hydrogens):
                atom_labels.append('H')

        #detect branches
        try:
            branch_segment=branches[now]
            for j in range(len(branch_segment)):
                SMILES,atom_labels=encoding_to_SMILES(branch_segment[j],branches,rings,atom_list,BO,fc,SMILES,now,atom_labels,EZcenters,RScenters)
        except KeyError: 
            pass
    SMILES+=')'
    return SMILES,atom_labels

