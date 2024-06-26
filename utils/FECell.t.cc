

namespace dftfe
{
  namespace utils
  {
    template <unsigned int dim>
    FECell<dim>::FECell(
      typename dealii::DoFHandler<dim>::active_cell_iterator dealiiFECellIter,
      const dealii::FiniteElement< dim, dim > &fe):
      d_mappingQ1(),
      d_feCell(fe)
    {
      this->reinit(dealiiFECellIter);
    }

    template <unsigned int dim>
    void FECell<dim>::reinit(
      typename dealii::DoFHandler<dim>::active_cell_iterator dealiiFECellIter)
    {
      d_dealiiFECellIter = dealiiFECellIter;

      auto bb = this->getBoundingBox();
      d_lowerLeft.resize(dim,0.0);
      d_upperRight.resize(dim,0.0);
      for(unsigned int j = 0; j < dim;j++)
        {
          d_lowerLeft[j] = bb.first[j];
          d_upperRight[j] = bb.second[j];
        }
    }

    template <unsigned int dim>
    void
    FECell<dim>::getVertices(std::vector<std::vector<double>> & points) const
    {
      const unsigned int nVertices =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      points.resize(nVertices, std::vector<double>(dim));
      std::vector<dealii::Point<dim, double>> pointsDealii;
      pointsDealii.resize(nVertices);
      for (unsigned int iVertex = 0; iVertex < nVertices; iVertex++)
        {
          pointsDealii[iVertex] = d_dealiiFECellIter->vertex(iVertex);
          for(unsigned int j = 0;  j < dim; j++)
            {
              points[iVertex][j] =  pointsDealii[iVertex][j];
            }
        }
    }

    template <unsigned int dim>
    void
    FECell<dim>::getVertex(size_type i, std::vector<double> &point) const
    {
      point.resize(dim);
      dealii::Point<dim, double> pointDealii = d_dealiiFECellIter->vertex(i);
      for(unsigned int j = 0;  j < dim; j++)
        {
          point[j] =  pointDealii[j];
        }
    }

    template <unsigned int dim>
    std::pair<std::vector<double>, std::vector<double>>
    FECell<dim>::getBoundingBox() const
    {
      std::vector<double> ll(dim,0.0), ur(dim,0.0);
      dealii::BoundingBox<dim> bb = d_dealiiFECellIter->bounding_box();
      auto dealiiPointsPair = bb.get_boundary_points();
      for(unsigned int j = 0;  j < dim; j++)
        {
          ll[j] = (dealiiPointsPair.first)[j];
          ur[j] = (dealiiPointsPair.second)[j];
        }
      auto returnVal = make_pair(ll,ur);
      return returnVal;
    }

    template <unsigned int dim>
    bool
    FECell<dim>::isPointInside(const std::vector<double> &point,
                               const double tol) const
    {
      bool returnVal = true;
      std::vector<double> paramPoint = this->getParametricPoint(point);
      for( unsigned int j = 0 ; j< dim; j++)
        {
          if((paramPoint[j] < -tol) || (paramPoint[j] > 1.0 + tol))
            {
              returnVal = false;
            }
        }
      return returnVal;
    }

    template <unsigned int dim>
    std::vector<double>
    FECell<dim>::getParametricPoint(const std::vector<double> &realPoint)const
    {
      dealii::Point<dim, double> pointRealDealii;
      for(unsigned int j = 0 ; j <dim; j++)
        {
          pointRealDealii[j] = realPoint[j];
        }
      dealii::Point<dim, double> pointParamDealii =
        d_mappingQ1.transform_real_to_unit_cell(d_dealiiFECellIter,pointRealDealii);

      std::vector<double> pointParam(dim,0.0);
      for(unsigned int j = 0 ; j <dim; j++)
        {
          pointParam[j] = pointParamDealii[j];
        }

      return pointParam;
    }

    template <unsigned int dim>
    void FECell<dim>::
      getShapeFuncValues(unsigned int numPointsInCell,
                       const std::vector<double> &coordinatesOfPointsInCell,
                       std::vector<dataTypes::number> &shapeFuncValues,
                       unsigned int cellShapeFuncStartIndex,
                       unsigned int numDofsPerElement) const
    {
              for( size_type iPoint = 0 ;iPoint < numPointsInCell; iPoint++)
                {

                  dealii::Point<3, double> realCoord(coordinatesOfPointsInCell[3*iPoint+0],
                                                          coordinatesOfPointsInCell[3*iPoint+1],
                                                          coordinatesOfPointsInCell[3*iPoint+2]);

                  dealii::Point<dim, double> pointParamDealii =
                    d_mappingQ1.transform_real_to_unit_cell(d_dealiiFECellIter,realCoord);

                  AssertThrow((pointParamDealii[0] > -1e-7) && (pointParamDealii[0] < 1 +  1e-7),
                              dealii::ExcMessage("param point x coord is -ve\n"));
                  AssertThrow((pointParamDealii[1] > -1e-7) && (pointParamDealii[1] < 1 +  1e-7),
                              dealii::ExcMessage("param point y coord is -ve\n"));
                  AssertThrow((pointParamDealii[2] > -1e-7) && (pointParamDealii[2] < 1 +  1e-7),
                              dealii::ExcMessage("param point z coord is -ve\n"));

                  for (unsigned int iNode = 0; iNode < numDofsPerElement;
                       iNode++)
                    {
                      shapeFuncValues[cellShapeFuncStartIndex +
                                        iNode +
                                        iPoint * numDofsPerElement] = d_feCell.shape_value(iNode, pointParamDealii);
                    }
                }
    }
  } // end of namespace utils

} // end of namespace dftfe
