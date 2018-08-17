// Geometric hashing example
// Obtained from http://www.cs.rpi.edu/academics/courses/spring08/cs2/homework/09/

#include "geo_hash.h"

#include <iostream>
#include <vector>

//  A simple stream operators to output a bin.
std::ostream& operator<< ( std::ostream& s, bin_index b )
{
  s << b.i() << ',' << b.j();
  return s;
}

//  A simple stream operators to output a point.
std::ostream& operator<< ( std::ostream& s, point p )
{
  s << p.x() << ',' << p.y();
  return s;
}


int main1()
{
  const float bw = 0.125;
  geo_hash h1( bw );

  //
  //  Just testing the point_to_bin and hash_value functions first,
  //  without actually doing any inserts.
  //  
  point p( 1,0 );
  bin_index bi = h1.point_to_bin( p );
  unsigned int hv = h1.hash_value( p );
  std::cout << "(1) point = " << p << ", bin_index = " << bi
            << ", hash_value = " << hv << '\n';

  p = point(-1,0);
  bi = h1.point_to_bin( p );
  hv = h1.hash_value( p );
  std::cout << "(2) point = " << p << ", bin_index = " << bi
            << ", hash_value = " << hv << '\n';

  p = point(0.25,-0.33);
  bi = h1.point_to_bin( p );
  hv = h1.hash_value( p );
  std::cout << "(3) point = " << p << ", bin_index = " << bi
            << ", hash_value = " << hv << '\n';
  std::cout << "--------------------\n";

  //
  //  The first four points actually added should all go into bin 2,2
  //
  point p1(  0,  0 );
  point p2( 60,  0 );
  point p3( 60, 80 );
  point p4(  0, 80 );
  std::vector<point> points_to_add;
  points_to_add.push_back(p2);
  points_to_add.push_back(p3);
  points_to_add.push_back(p4);
  h1.add_point( p1 );
  h1.add_points( points_to_add );
  std::vector< point > points = h1.points_in_bin( p4 );
  std::cout << "(4) The bin associated with " << p4 << " has "
            << points.size() << " point(s).  They are\n";
  for ( unsigned int i=0; i<points.size(); ++i )
    std::cout << "\t" << points[i] << '\n';
  std::cout << "(5) Number of non-empty bins = " << h1.num_non_empty() << '\n';
  std::cout << "(6) Number of points stored = " << h1.num_points() << '\n';
  std::cout << "(7) Table size = " << h1.table_size() << '\n';
  std::cout << "--------------------\n";

  //
  //  Add points in five more bins.  This should not cause a resize.
  //  
  h1.add_point( point(5,13) );
  h1.add_point( point(-4,9) );
  h1.add_point( point(-10,9) );
  h1.add_point( point(8,-8) );
  h1.add_point( point(20,-13) );
  h1.add_point( point(21,-17) );
  point p5( 20.5, -15 );
  points = h1.points_in_bin( p5 );
  std::cout << "(8) The bin associated with " << p5 << " has "
            << points.size() << " point(s).  They are\n";
  for ( unsigned int i=0; i<points.size(); ++i )
    std::cout << "\t" << points[i] << '\n';
  std::cout << "(9) Number of non-empty bins = " << h1.num_non_empty() << '\n';
  std::cout << "(10) Number of points stored = " << h1.num_points() << '\n';
  std::cout << "(11) Table size = " << h1.table_size() << '\n';
  std::cout << "--------------------\n";

  //  
  //  This next point added should cause a resizing of the table.
  //
  h1.add_point( point(2,-13.5) );
  std::cout << "(9) Number of non-empty bins = " << h1.num_non_empty() << '\n';
  std::cout << "(10) Number of points stored = " << h1.num_points() << '\n';
  std::cout << "(11) Table size = " << h1.table_size() << '\n';
  std::cout << "--------------------\n";

  //  
  //  Circle search when all the points are within one bin, but not
  //  all should be included.
  //
  point center( 8.75, 7.5 );
  float radius = 1.75;
  points = h1.points_in_circle( center, radius );
  std::cout << "(12) Within a circle centered at " << center << ", with radius "
            << radius << ", there are " << points.size() << " point(s).  They are\n";
  for ( unsigned int i=0; i<points.size(); ++i )
    std::cout << "\t" << points[i] << '\n';
  std::cout << "--------------------\n";

  //
  //  Circle search is now much bigger and involves several bins
  //  
  center = point( 1.1, 9.25 );
  radius = 6.5;
  points = h1.points_in_circle( center, radius );
  std::cout << "(13) Within a circle centered at " << center << ", with radius "
            << radius << ", there are " << points.size() << " point(s).  They are\n";
  for ( unsigned int i=0; i<points.size(); ++i )
    std::cout << "\t" << points[i] << '\n';
  std::cout << "--------------------\n";

  //
  //  This circle should be empty, even though it intersects bins
  //  having points.
  // 
  center = point( -3.9, -3 );
  radius = 6.25;
  points = h1.points_in_circle( center, radius );
  std::cout << "(14) Within a circle centered at " << center << ", with radius "
            << radius << ", there are " << points.size() << " point(s).\n";
  std::cout << "--------------------\n";

  //
  //  Erasing single point.  The number of bins should not decrease.
  //
  point p6(21,-17);
  int num_erased = h1.erase_points( p6 );
  std::cout << "(15) Erasing point " << p6 << ", num_erased = " << num_erased << '\n';
  std::cout << "(16) Number of non-empty bins = " << h1.num_non_empty() << '\n';
  std::cout << "(17) Number of points stored = " << h1.num_points() << '\n';
  std::cout << "--------------------\n";

  //
  //  Erasing in a larger circle.  Two points and two bins should go.
  //
  center = point(14,-10);
  radius = 8.5;
  num_erased = h1.erase_points( center, radius );
  std::cout << "(18) Erasing points in circle centered at " << center 
            << ", with radius = " << radius << "; num_erased = " 
            <<  num_erased << '\n';
  std::cout << "(19) Number of non-empty bins = " << h1.num_non_empty() << '\n';
  std::cout << "(20) Number of points stored = " << h1.num_points() << '\n';
  std::cout << "--------------------\n";


  //
  //  Putting values back, including at some formerly deleted bins.
  //  Table size should not change
  // 
  h1.add_point( point(22,-17.5) );
  h1.add_point( point(13,15) );
  h1.add_point( point(15.5,15.5) );
  h1.add_point( point(20, 15.3) );
  h1.add_point( point(21, 15.6) );
  h1.add_point( point(22, 17.6) );
  h1.add_point( point(19, 8.5) );
  h1.add_point( point(21, 4.6) );
  h1.add_point( point(21.3, 3.7) );
  h1.add_point( point(27.8, 5.7) );
  std::cout << "(21) Number of non-empty bins = " << h1.num_non_empty() << '\n';
  std::cout << "(22) Number of points stored = " << h1.num_points() << '\n';
  std::cout << "(23) Table size = " << h1.table_size() << '\n';
  std::cout << "--------------------\n";

  //
  //  Rectangle search starting with two empty ones
  //
  point p_min( 123.1, -19 );
  point p_max( 155.1, 11 );
  points = h1.points_in_rectangle( p_min, p_max );
  std::cout << "(24) Rectangle with corners " << p_min << " and " << p_max
            << ", contains " << points.size() << " point(s).\n";
  p_min = point( -8, -8 );
  p_max = point( 7,7 );
  points = h1.points_in_rectangle( p_min, p_max );
  std::cout << "(25) Rectangle with corners " << p_min << " and " << p_max
            << ", contains " << points.size() << " point(s).\n";
  std::cout << "--------------------\n";

  //
  //  Rectangle search involving only one square
  //
  p_min = point( 9.3, 8.9 );
  p_max = point( 9.6, 9.1 );
  points = h1.points_in_rectangle( p_min, p_max );
  std::cout << "(26) Rectangle with corners " << p_min << " and " << p_max
            << ", contains " << points.size() << " point(s).  Included are\n";
  for ( unsigned int i=0; i<points.size(); ++i )
    std::cout << "\t" << points[i] << '\n';
  std::cout << "--------------------\n";
  
  //
  //  Rectangle search involving multiple squares
  //
  p_max = point( 23,17 );
  std::cout << "(27) Rectangle with corners " << p_min << " and " << p_max
            << ", contains " << points.size() << " point(s).  Included are\n";
  points = h1.points_in_rectangle( p_min, p_max );
  for ( unsigned int i=0; i<points.size(); ++i )
    std::cout << "\t" << points[i] << '\n';
  std::cout << "--------------------\n";
  std::cout << "--------------------\n";


  //
  //  Just to be sure, here is a new geometric hash table with a different bin width
  //
  const float new_width = 18;
  geo_hash h2( new_width );

  //
  //  Repeating some tests on the bin and hash function without
  //  actually doing any inserts.
  //  
  p = point( 8,10 );
  bi = h2.point_to_bin( p );
  hv = h2.hash_value( p );
  std::cout << "(28) point = " << p << ", bin_index = " << bi
            << ", hash_value = " << hv << '\n';
  p = point( -54, -21 );
  bi = h2.point_to_bin( p );
  hv = h2.hash_value( p );
  std::cout << "(29) point = " << p << ", bin_index = " << bi
            << ", hash_value = " << hv << '\n';

  points.clear();
  points.push_back( point(35,21) );
  points.push_back( point(-45,15) );
  points.push_back( point(-16,18) );
  points.push_back( point(-12,55) );
  points.push_back( point(-17,18) );
  points.push_back( point(-81,21) );
  points.push_back( point(-87,22) );
  points.push_back( point(-89.5,29) );
  points.push_back( point(-88.5,20) );
  points.push_back( point(-81,-35) );
  h2.add_points( points );

  p = point(-87,28);
  points = h2.points_in_bin( p );
  std::cout << "(30) The bin associated with " << p << " has "
            << points.size() << " point(s).  They are\n";
  for ( unsigned int i=0; i<points.size(); ++i )
    std::cout << "\t" << points[i] << '\n';
  std::cout << "(31) Number of non-empty bins = " << h2.num_non_empty() << '\n';
  std::cout << "(32) Number of points stored = " << h2.num_points() << '\n';
  std::cout << "(33) Table size = " << h2.table_size() << '\n';
  std::cout << "--------------------\n";
  

  return 0;
}
