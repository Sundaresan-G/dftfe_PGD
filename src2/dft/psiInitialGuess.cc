#include <fstream>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>

//load PSI radial value files
void dftClass::loadPSIFiles(unsigned int Z, 
			    unsigned int n, 
			    unsigned int l,
			    unsigned int &fileReadFlag)
{
  //
  if (radValues[Z][n].count(l)>0) 
    {
      fileReadFlag = 1;
      return;
    }
  //
  char psiFile[256];
  if(isPseudopotential)
    sprintf(psiFile, "../../../data/electronicStructure/pseudoPotential/z%u/SingleAtomData/psi%u%u.inp", Z, n, l);
  else
    sprintf(psiFile, "../../../data/electronicStructure/allElectron/z%u/psi%u%u.inp", Z, n, l);
  std::vector<std::vector<double> > values;

  fileReadFlag = readPsiFile(2, values, psiFile);
  

  if(fileReadFlag > 0)
    {
      pcout<<"Reading data from file: "<<psiFile<<std::endl;
      
      int numRows = values.size()-1;
      std::vector<double> xData(numRows), yData(numRows);

      //x
      for(int irow = 0; irow < numRows; ++irow)
	{
	  xData[irow]= values[irow][0];
	}
      outerValues[Z][n][l] = xData[numRows-1];
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);	

      //y
      for(int irow = 0; irow < numRows; ++irow)
	{
	  yData[irow] = values[irow][1];
	}
      alglib::real_1d_array y;
      y.setcontent(numRows,&yData[0]);
      alglib::ae_int_t natural_bound_type = 0;
      alglib::spline1dinterpolant* spline=new alglib::spline1dinterpolant;
      alglib::spline1dbuildcubic(x, y, numRows,
				 natural_bound_type,
				 0.0,
				 natural_bound_type,
				 0.0,
				 *spline);
      //pcout << "send: Z:" << Z << " n:" << n << " l:" << l << " numRows:" << numRows << std::endl; 
      radValues[Z][n][l]=spline;
    }
}

//determine orbital ordering
void dftClass::determineOrbitalFilling()
{
  //create stencil following orbital filling order
  std::vector<unsigned int> level;
  std::vector<std::vector<unsigned int> > stencil;
  //1s
  level.clear(); level.push_back(1); level.push_back(0); stencil.push_back(level);
  //2s
  level.clear(); level.push_back(2); level.push_back(0); stencil.push_back(level);
  //2p
  level.clear(); level.push_back(2); level.push_back(1); stencil.push_back(level);
  //3s
  level.clear(); level.push_back(3); level.push_back(0); stencil.push_back(level);
  //3p
  level.clear(); level.push_back(3); level.push_back(1); stencil.push_back(level);
  //4s
  level.clear(); level.push_back(4); level.push_back(0); stencil.push_back(level);
  //3d
  level.clear(); level.push_back(3); level.push_back(2); stencil.push_back(level);
  //4p
  level.clear(); level.push_back(4); level.push_back(1); stencil.push_back(level);
  //5s
  level.clear(); level.push_back(5); level.push_back(0); stencil.push_back(level);
  //4d
  level.clear(); level.push_back(4); level.push_back(2); stencil.push_back(level);
  //5p
  level.clear(); level.push_back(5); level.push_back(1); stencil.push_back(level);
  //6s
  level.clear(); level.push_back(6); level.push_back(0); stencil.push_back(level);
  //4f
  level.clear(); level.push_back(4); level.push_back(3); stencil.push_back(level);
  //5d
  level.clear(); level.push_back(5); level.push_back(2); stencil.push_back(level);
  //6p
  level.clear(); level.push_back(6); level.push_back(1); stencil.push_back(level);
  //7s
  level.clear(); level.push_back(7); level.push_back(0); stencil.push_back(level);
  //5f
  level.clear(); level.push_back(5); level.push_back(3); stencil.push_back(level);
  //6d
  level.clear(); level.push_back(6); level.push_back(2); stencil.push_back(level);
  //7p
  level.clear(); level.push_back(7); level.push_back(1); stencil.push_back(level);
  //8s
  level.clear(); level.push_back(8); level.push_back(0); stencil.push_back(level);
  
  int totalNumberWaveFunctions = numEigenValues;
  unsigned int fileReadFlag = 0;
  unsigned int waveFunctionCount = 0;
  
  //loop over atoms
  for (unsigned int z = 0; z < atomLocations.size(); z++)
    {
      unsigned int Z = atomLocations[z][0];
      unsigned int valenceZ = atomLocations[z][1];
        
      //check if additional wave functions requested
      //unsigned int additionalLevels=0;
      //if (additionalWaveFunctions.count(Z)!=0) {
      //additionalLevels=additionalWaveFunctions[Z];
      //} 
      unsigned int totalLevels= numberAtomicWaveFunctions[Z];//((unsigned int)std::ceil(Z/2.0))+additionalLevels;
      if(isPseudopotential)
	numElectrons+=valenceZ;
      else
	numElectrons+=Z;
      //numBaseLevels+=(unsigned int)std::ceil(Z/2.0);
      numLevels+=totalLevels;
    
      //fill levels
      bool printLevels=false;
      if (radValues.count(Z)==0){
	printLevels=true;
	pcout << "Z:" << Z << std::endl;
      }
      unsigned int levels=0;
      for (std::vector<std::vector<unsigned int> >::iterator it=stencil.begin(); it <stencil.end(); it++)
	{
	  unsigned int n=(*it)[0], l=(*it)[1];
	  //load PSI files
	  loadPSIFiles(Z, n, l,fileReadFlag);
	  //m loop
	  if(fileReadFlag > 0)
	    {
	      for (int m=-l; m<= (int) l; m++)
		{
		  orbital temp;
		  temp.atomID=z;
		  temp.Z=Z; temp.n=n; temp.l=l; temp.m=m; temp.psi=radValues[Z][n][l];
		  waveFunctionsVector.push_back(temp); levels++; waveFunctionCount++;
		  if(printLevels) pcout << " n:" << n  << " l:" << l << " m:" << m << std::endl;
		  if(levels >= totalLevels || waveFunctionCount>=numEigenValues) break;
		}
	    }
	  if(levels>=totalLevels || waveFunctionCount>=numEigenValues) break;
	}
    }
  pcout<<"*****************************************************************************************************************"<<std::endl;
  pcout<<"Total number electrons: "<<numElectrons<<std::endl;
  pcout<<"Total number of wavefunctions as initial guess for eigensolver computed using single atom wavefunctions: " <<waveFunctionCount<<std::endl;
  pcout<<"*****************************************************************************************************************"<<std::endl;
}

//
void dftClass::readPSIRadialValues(){
  //loop over nodes to set PSI initial guess
  //get support points
  //std::map<types::global_dof_index, Point<3> > support_points;
  //MappingQ<3> mapQ(1);
  //DoFTools::map_dofs_to_support_points(mapQ, dofHandler, support_points); 
  IndexSet locallyOwnedSet;
  DoFTools::extract_locally_owned_dofs(dofHandlerEigen, locallyOwnedSet);
  std::vector<unsigned int> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  std::vector<std::vector<double> > local_dof_values(numEigenValues, std::vector<double>(locallyOwnedDOFs.size(), 0.0));

#ifdef ENABLE_PERIODIC_BC
  unsigned int numberDofs = locallyOwnedDOFs.size()/2;
#else
  unsigned int numberDofs = locallyOwnedDOFs.size();
#endif
  
  //loop over nodes
  bool pp=false;
  for(unsigned int dof=0; dof<numberDofs; dof++)
    {
#ifdef ENABLE_PERIODIC_BC
      unsigned int dofID = locallyOwnedDOFs[2*dof];
#else
      unsigned int dofID= locallyOwnedDOFs[dof];
#endif
      Point<3> node = d_supportPointsEigen[dofID];
      //loop over wave functions
      unsigned int waveFunction=0;
      for (std::vector<orbital>::iterator it=waveFunctionsVector.begin(); it<waveFunctionsVector.end(); it++){
	//find coordinates of atom correspoding to this wave function

	Point<3> atomCoord(atomLocations[it->atomID][2],atomLocations[it->atomID][3],atomLocations[it->atomID][4]);
	//
	double x =node[0]-atomCoord[0];
	double y =node[1]-atomCoord[1];
	double z =node[2]-atomCoord[2];
	//
	double r = sqrt(x*x + y*y + z*z);
	double theta = acos(z/r);
	double phi = atan2(y,x);
	//
	if (r==0){theta=0; phi=0;}
	//radial part
	double R=0.0;
	if (r<=outerValues[it->Z][it->n][it->l]) R = alglib::spline1dcalc(*(it->psi),r);
	if (!pp){
	  //pcout << "atom: " << it->atomID << " Z:" << it->Z << " n:" << it->n << " l:" << it->l << " m:" << it->m << " x:" << atomCoord[0] << " y:" << atomCoord[1] << " z:" << atomCoord[2] << " Ro:" << outerValues[it->Z][it->n][it->l] << std::endl; 
	}

#ifdef ENABLE_PERIODIC_BC
	if (it->m > 0){
	  local_dof_values[waveFunction][2*dof] =  R*std::sqrt(2)*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
	}
	else if (it->m == 0){
	  local_dof_values[waveFunction][2*dof] =  R*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
	}
	else{
	  local_dof_values[waveFunction][2*dof] =  R*std::sqrt(2)*boost::math::spherical_harmonic_i(it->l,-(it->m),theta,phi);	  
	}
#else	
	//spherical part
	if (it->m > 0){
	  local_dof_values[waveFunction][dof] =  R*std::sqrt(2)*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
	}
	else if (it->m == 0){
	  local_dof_values[waveFunction][dof] =  R*boost::math::spherical_harmonic_r(it->l,it->m,theta,phi);
	}
	else{
	  local_dof_values[waveFunction][dof] =  R*std::sqrt(2)*boost::math::spherical_harmonic_i(it->l,-(it->m),theta,phi);	  
	}
#endif
	waveFunction++;
      }
      pp=true;
    }

  if(waveFunctionsVector.size() < numEigenValues)
    {
      unsigned int nonAtomicWaveFunctions = numEigenValues - waveFunctionsVector.size();
      pcout << "Levels Generated Randomly: " << nonAtomicWaveFunctions << std::endl;
      //
      // assign the rest of the wavefunctions using a standard normal distribution
      //
      boost::math::normal normDist;

      for(unsigned int iWave = waveFunctionsVector.size(); iWave < numEigenValues; ++iWave)
	{
	  for(unsigned int dof=0; dof<numberDofs; dof++)
	    {
	      double z = (-0.5 + (rand()+ 0.0)/(RAND_MAX))*3.0;
	      double value =  boost::math::pdf(normDist, z); 
	      if(rand()%2 == 0)
		value = -1.0*value;
	      local_dof_values[iWave][2*dof] = value;
	    }
	}

    }

  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      for (unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  constraintsNoneEigen.distribute_local_to_global(local_dof_values[i], locallyOwnedDOFs, *eigenVectors[kPoint][i]);
	}
    }
  //multiply by M^0.5
  for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      for (unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  for (unsigned int j = 0; j < eigenVectors[kPoint][i]->local_size(); j++)
	    {
	      if (std::abs(eigen.massVector.local_element(j))>1.0e-15)
		{
		  eigenVectors[kPoint][i]->local_element(j)/=eigen.massVector.local_element(j);
		}
	    }
	  char buffer[100];
	  sprintf(buffer, "norm %u: l1: %14.8e  l2:%14.8e\n",i, eigenVectors[kPoint][i]->l1_norm(), eigenVectors[kPoint][i]->l2_norm());
	  pcout << buffer;
	  eigenVectors[kPoint][i]->update_ghost_values();
	}
    }
}

//
void dftClass::readPSI(){
  computing_timer.enter_section("dftClass init PSI"); 
  //
  readPSIRadialValues();
  //
  computing_timer.exit_section("dftClass init PSI"); 
}
