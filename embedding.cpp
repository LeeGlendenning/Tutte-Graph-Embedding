#include <iostream>
#include <vector>
#include <math.h>
// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  MyMesh;
// -------------------- Eigen
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

MyMesh  mesh;

/*
	countVertices: counts and returns total number of vertices in mesh
*/
int countVertices() {

	int totalVerts = 0;
	// Loop through and count all vertices in mesh *v_it
	for (MyMesh::VertexIter vertex_iter = mesh.vertices_begin(); vertex_iter != mesh.vertices_end(); ++vertex_iter) {
		totalVerts++;
	}
	
	return totalVerts;
}


void uniformLaplacian(Eigen::SparseMatrix<double, Eigen::ColMajor> &a);
void laplaceBeltrami(Eigen::SparseMatrix<double, Eigen::ColMajor> &a);
void meanValue(Eigen::SparseMatrix<double, Eigen::ColMajor> &a);


/*
	fillMatrix: fill matrix a using specified weighting
*/
void fillMatrix(Eigen::SparseMatrix<double, Eigen::ColMajor> &a, std::string weight) {

	std::cout << weight << std::endl;

	if (weight == "Uniform-Laplacian") {
		uniformLaplacian(a);
	}
	else if (weight == "Laplace-Beltrami") {
		laplaceBeltrami(a);
	}
	else if (weight == "Mean-value") {
		meanValue(a);
	}
	else {
		std::cout << "Invalid weight" << std::endl;
		return;
	}

	
}


void uniformLaplacian(Eigen::SparseMatrix<double, Eigen::ColMajor> &a) {

	// Uniform laplacian weighting: fill matrix with 1's if an edge exists
	// Loop through all vertices in mesh *v_it
	for (MyMesh::VertexIter vertex_iter = mesh.vertices_begin(); vertex_iter != mesh.vertices_end(); ++vertex_iter) {
		int v_index = vertex_iter.handle().idx();

		if (mesh.is_boundary(*vertex_iter)) {
			a.insert(v_index, v_index) = 1.0;
		}
		else {
			// Set weighting for matrix diagonals to be -1*(#neighbours)
			a.insert(v_index, v_index) = -(double)mesh.valence(*vertex_iter);

			// Iterate through all neighbours (*vv_it) to current vertex *v_it
			for (MyMesh::VertexVertexIter neighbour_iter = mesh.vv_iter(*vertex_iter); neighbour_iter.is_valid(); ++neighbour_iter) {
				if (!mesh.is_boundary(*neighbour_iter)) {
					a.insert(v_index, neighbour_iter.handle().idx()) = 1.0;
				}
			}
		}
	}
}


double cotan(double x) { 
	return(1 / tan(x)); 
}


double getVoronoiArea(MyMesh::VertexIter vertex_iter) {
	//std::cout << "-------Finding voronoi area-------" << std::endl;
	double voronoiArea = 0.0;
	// Calculate the voronoi area for vertex
	// for each adjacent face
	for (MyMesh::VertexFaceIter vface_iter = mesh.vf_iter(*vertex_iter); vface_iter.is_valid(); ++vface_iter)
	{
		int uvIndex = 0;
		double uvLength[2]; // Lengths of edges u and v
		double alpha, beta; // Angles opposite i

		// Get adjacent edge lengths u and v to i
		for (MyMesh::FaceHalfedgeIter fedge_iter = mesh.fh_iter(*vface_iter); fedge_iter.is_valid(); ++fedge_iter)
		{
			// The 2 vertices defining the current edge
			MyMesh::Point to = mesh.point(mesh.to_vertex_handle(fedge_iter.handle()));
			MyMesh::Point from = mesh.point(mesh.from_vertex_handle(fedge_iter.handle()));
			// If to or from is i, the other is u or v
			if (to == mesh.point(*vertex_iter) || from == mesh.point(*vertex_iter)) {
				if (from == mesh.point(*vertex_iter)) {
					alpha = mesh.calc_sector_angle(fedge_iter.handle());
				}
				uvLength[uvIndex] = mesh.calc_edge_length(fedge_iter.handle());
				uvIndex++;
			}
			else {
				beta = mesh.calc_sector_angle(fedge_iter.handle());
			}
		}

		//		this is the voronoi area of an adjacent face to vertex i
		//		voronoiArea += (1/8)(u^2 * cot(alpha) + v^2 * cot(beta))
		//		where, u and v are adjacent edges in the face,
		//		alpha and beta are the opposite angles of u and v respectively
		voronoiArea += (1.0 / 8.0)*(pow(uvLength[0], 2) * cotan(alpha) + pow(uvLength[1], 2) * cotan(beta));
	}

	return voronoiArea;
}


void laplaceBeltrami(Eigen::SparseMatrix<double, Eigen::ColMajor> &a) {

	// Laplace-Beltrami weighting: 
	// Loop through and count all vertices in mesh *v_it
	for (MyMesh::VertexIter vertex_iter = mesh.vertices_begin(); vertex_iter != mesh.vertices_end(); ++vertex_iter) {
		int i_index = vertex_iter.handle().idx();

		if (mesh.is_boundary(*vertex_iter)) {
			a.insert(i_index, i_index) = 1.0;
		}
		else {
			double voronoiArea = getVoronoiArea(vertex_iter);
			
			// Set weighting for matrix diagonals to be -1*(#neighbours)
			a.insert(i_index, i_index) = -(double)mesh.valence(*vertex_iter);
			
			MyMesh::VertexOHalfedgeIter neighbour_iter = mesh.voh_iter(*vertex_iter);
			MyMesh::HalfedgeHandle prevj = neighbour_iter.handle();
			neighbour_iter++;
			
			// For each neighbour j of i
			for (neighbour_iter; neighbour_iter.is_valid(); ++neighbour_iter) {
				// neighbour_iter points to j
				if (!mesh.is_boundary(mesh.to_vertex_handle(neighbour_iter))) {
					double alpha = mesh.calc_sector_angle(prevj);
					// betaEdge =  half edge defined by j and *neighbour_iter
					// i.e. since *neighbour_iter points to j (outgoing from i), the next half edge is beta edge
					double beta = mesh.calc_sector_angle(mesh.next_halfedge_handle(*neighbour_iter));
					//double beta = mesh.calc_sector_angle(mesh.find_halfedge(mesh.to_vertex_handle(j), mesh.to_vertex_handle(mesh.halfedge_handle(neighbour_iter, 0))));

					// Alpha and beta are the angles of the vertices adjacent to both i and j
					double weight = (1 / (2 * voronoiArea)) * (cotan(alpha) + cotan(beta));

					int j_index = mesh.to_vertex_handle(neighbour_iter.handle()).idx();
					a.insert(i_index, j_index) = weight;
				}
				prevj = neighbour_iter.handle();
			}


		}
	}
}


double euclideanDist(MyMesh::Point p, MyMesh::Point q) {
	return sqrt(pow(p[0] - q[0], 2) + pow(p[1] - q[1], 2));
}


void meanValue(Eigen::SparseMatrix<double, Eigen::ColMajor> &a) {
	// Mean value weighting: 
	// Loop through and count all vertices in mesh *v_it
	for (MyMesh::VertexIter vertex_iter = mesh.vertices_begin(); vertex_iter != mesh.vertices_end(); ++vertex_iter) {
		int i_index = vertex_iter.handle().idx();

		if (mesh.is_boundary(*vertex_iter)) {
			a.insert(i_index, i_index) = 1.0;
		}
		else {
			// Set weighting for matrix diagonals to be -1*(#neighbours)
			a.insert(i_index, i_index) = -(double)mesh.valence(*vertex_iter);

			MyMesh::VertexIHalfedgeIter edge_iter = mesh.vih_iter(*vertex_iter);
			MyMesh::HalfedgeHandle prevj = edge_iter.handle();
			edge_iter++;

			// if prevj's next half edge has i as from
			// this doesn't seem to happen
			if (mesh.from_vertex_handle(mesh.next_halfedge_handle(prevj)) != vertex_iter.handle() && edge_iter.is_valid()) {
				std::cout << "skipping prevj, need next: " << std::endl;
				prevj = edge_iter;
				edge_iter++;
			}

			// Iterate through all adjacent in halfedges
			for (edge_iter; edge_iter.is_valid(); ++edge_iter) {
				if (!mesh.is_boundary(mesh.from_vertex_handle(edge_iter))) {
					double delta, gamma;
					// Use prevj half edge to find delta angle
					delta = mesh.calc_sector_angle(prevj);
					
					// Use j half edge to find gamma angle
					gamma = mesh.calc_sector_angle(mesh.next_halfedge_handle(*edge_iter));
					
					double weight = (1 / euclideanDist(mesh.point(*vertex_iter), mesh.point(mesh.from_vertex_handle(edge_iter)))) * (tan(delta / 2) + tan(gamma / 2));
					int j_index = mesh.from_vertex_handle(edge_iter).idx();
					a.insert(i_index, j_index) = weight;
				}
				prevj = edge_iter.handle();
			}
		}

	}
}


void fillVectors(Eigen::VectorXd &b, Eigen::VectorXd &d) {
	MyMesh::VertexIter vertex_iter, v_end(mesh.vertices_end());

	// Loop through all vertices in mesh
	// For each boundary point, put x coord in b and y coord in d, otherwise 0
	for (vertex_iter = mesh.vertices_begin(); vertex_iter != v_end; ++vertex_iter) {
		int v_index = vertex_iter.handle().idx();

		if (mesh.is_boundary(*vertex_iter)) {
			b(v_index) = mesh.point(*vertex_iter)[0];
			d(v_index) = mesh.point(*vertex_iter)[1];
		}
		else {
			b(v_index) = 0.0;
			d(v_index) = 0.0;
		}
	}
}


int solveLinearSystems(Eigen::SparseMatrix<double, Eigen::ColMajor> a, Eigen::VectorXd b, Eigen::VectorXd d, Eigen::VectorXd &x, Eigen::VectorXd &y) {
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double, Eigen::ColMajor> >   solverA;
	a.makeCompressed();
	solverA.analyzePattern(a);
	solverA.factorize(a);
	if (solverA.info() != Eigen::Success) {
		std::cout << "Error: Decomposition A failed" << "\n";
		return 1;
	}

	std::cout << "Factorized a" << std::endl;
	x = solverA.solve(b);
	if (solverA.info() != Eigen::Success) {
		std::cout << "Error: Solving b failed" << "\n";
		return 1;
	}
	std::cout << "Solved x" << std::endl;

	y = solverA.solve(d);
	if (solverA.info() != Eigen::Success) {
		std::cout << "Error: Solving d failed" << "\n";
		return 1;
	}
	std::cout << "Solved y" << std::endl;
	return 0;
}


void updateVertexPositions(Eigen::VectorXd x, Eigen::VectorXd y) {
	MyMesh::VertexIter vertex_iter, v_end(mesh.vertices_end());

	// Loop through all vertices *v_it in mesh 
	//int i = 0, j = 0;
	for (vertex_iter = mesh.vertices_begin(); vertex_iter != v_end; ++vertex_iter) {
		// If vertex is not on boundary
		//if (!mesh.is_boundary(*vertex_iter)) {
		if (x(vertex_iter.handle().idx()) != 0){
			//std::cout << "x[" << vertex_iter.handle().idx() << "] = " << x(vertex_iter.handle().idx()) << ", " << y(vertex_iter.handle().idx()) << std::endl;;
			
			mesh.point(*vertex_iter)[0] = x(vertex_iter.handle().idx());
			mesh.point(*vertex_iter)[1] = y(vertex_iter.handle().idx());

		}
		mesh.point(*vertex_iter)[2] = 0;

	}
}

// Save paramaterization as texture coordinates of original mesh
/*
> the per face texture coordinates are stored at the halfedges. As we have
> per face per vertex one halfedge, the coordinates are associated with
> the halfedges. The texcoordinate at the halfedge is the one belonging to
> the adjacent face and the to vertex of the he.
*/
void setTextureCoordinates(Eigen::VectorXd x, Eigen::VectorXd y)
{
	mesh.request_vertex_texcoords3D();
	//MyMesh::HalfedgeHandle heh, heh_init;
	//MyMesh::HalfedgeHandle heh_orig, heh_init_orig;

	// Get first halfedge handle for original mesh and working mesh
	/*heh_orig = heh_init_orig = origMesh.halfedges_begin();
	heh = heh_init = mesh.halfedges_begin();
	// Iterate through all halfedges and set texture coordinates accordingly
	do {
		//std::cout << heh_orig.idx() << ", " << origMesh.has_halfedge_texcoords3D() << std::endl;
		origMesh.set_texcoord3D(heh_orig, mesh.texcoord3D(heh));
		heh_orig = origMesh.next_halfedge_handle(heh_orig);
		heh = mesh.next_halfedge_handle(heh);
	}while (heh != heh_init && heh_orig != heh_init_orig);*/
	
	

	MyMesh::VertexIter vertex_iter, v_end(mesh.vertices_end());
	for (vertex_iter = mesh.vertices_begin(); vertex_iter != v_end; ++vertex_iter) 
	{
		if (!mesh.is_boundary(*vertex_iter)) {
			//std::cout << mesh.point(*vertex_iter) << " | " << x(vertex_iter.handle().idx()) << ", " << y(vertex_iter.handle().idx()) << std::endl;
			mesh.set_texcoord3D(*vertex_iter, { x(vertex_iter.handle().idx()), y(vertex_iter.handle().idx()), 0 });
		}
		mesh.point(*vertex_iter)[2] = 0;
	}

}


int main(int argc, char **argv)
{
	// check command line options
	if (argc != 4)
	{
		std::cerr << "Usage:  " << argv[0] << " <input_file> <output_file> <weight>\n";
		return 1;
	}
	// read mesh from input_file argv[1]
	if (!OpenMesh::IO::read_mesh(mesh, argv[1]))
	{
		std::cerr << "Error: Cannot read mesh from " << argv[1] << std::endl;
		return 1;
	}
	

	MyMesh::VertexIter          vertex_iter, v_end(mesh.vertices_end());
	MyMesh::VertexVertexIter    neighbour_iter;

	int totalVerts = countVertices();
	std::cout << "Total vertices in model: " << totalVerts << std::endl; // 5000

	Eigen::SparseMatrix<double, Eigen::ColMajor> a(totalVerts, totalVerts);
	a.reserve(totalVerts);
	Eigen::VectorXd b(totalVerts);
	Eigen::VectorXd d(totalVerts);
	Eigen::VectorXd x(totalVerts);
	Eigen::VectorXd y(totalVerts);

	fillMatrix(a, argv[3]);

	fillVectors(b, d);
	
	// Solve linear systems of equations
	solveLinearSystems(a, b, d, x, y);
	
	// Update vertex positions with new x and y values and set z=0
	updateVertexPositions(x, y);

	setTextureCoordinates(x, y);

	// Write mesh to output_file argv[2]
	if (!OpenMesh::IO::write_mesh(mesh, argv[2])) {
		std::cerr << "Error: cannot write mesh to " << argv[2] << std::endl;
		return 1;
	}

	return 0;
}
