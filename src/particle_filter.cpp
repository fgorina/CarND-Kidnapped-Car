/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"


using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);
   
    // Generate particles
    num_particles = 100;
    for(int i = 0; i < num_particles; i++){
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }
    
    if(debug){
        checkParticles("Generating", x, y, theta, std);
    }
    is_initialized = true;
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    
    for(int i = 0; i < particles.size(); i++){
        
        Particle p = particles[i];
        double x0 = p.x;
        double y0 = p.y;
        double theta0 = p.theta;
        
        // Compute new position
        // TODO: Add a check if yaw rate == 0 then update is different
        double dx;
        double dy;
        
        if (fabs(yaw_rate) > 0.001){
            dx = velocity / yaw_rate * (-sin(theta0)+sin(theta0 + delta_t*yaw_rate));
            dy = velocity / yaw_rate * (cos(theta0)-cos(theta0 + delta_t*yaw_rate));
        }else{
            dx = velocity * delta_t * cos(theta0);
            dy = velocity * delta_t * sin(theta0);
        }
        particles[i].x = x0 + dx;
        particles[i].y = y0 + dy;
        particles[i].theta = theta0 + yaw_rate*delta_t;
        
        // Add noise
        
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    
        if (isnan(particles[i].x) || isnan(particles[i].y)){
            cout << "Something went wrong" << endl;
        }
        // Normalize theta
        
        //particles[i].theta = Normalize( particles[i].theta);
    }

    if(debug){
        checkParticles("Prediction", 0, 0, 0, std_pos);
    }
}

/* What we really do is otherwise as stated. We assign the landmark to the observation. Is clearer to my brain. We change the car measures to the map ones
 
 This is different as stated in the original code as I translate observations to map coordinates.
 
     Reason is that we shhould have less observations than landmarks each time
     so number of transformations is smaller.
 
*/

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> map_landmarks, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (int iobs = 0; iobs < observations.size(); iobs++){
        
        LandmarkObs o = observations[iobs];
        
        int minl = -1;
        double mind = 1000000000;
        
        for(int iland = 0; iland < map_landmarks.size(); iland++){
            
            Map::single_landmark_s lo = map_landmarks[iland];
            double d = dist(o.x, o.y, lo.x_f, lo.y_f);
            if(d < mind){
                minl = iland;
                mind = d;
            }
        }
        
        observations[iobs].id = minl;
    }
}

/* A bit different as I convert everything to map coordinates */

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    double sigmax = std_landmark[0];
    double sigmay = std_landmark[1];

    
    for(int ip = 0; ip < particles.size(); ip++){

        vector<LandmarkObs> mapObservations;

        Particle p = particles[ip];
    
        for(int i = 0; i < observations.size(); i++){
            double map_x;
            double map_y;
            
            LandmarkObs lo = observations[i];
            std::tie(map_x, map_y) = transform(p, lo.x, lo.y);
            
            LandmarkObs ln;
            ln.id = lo.id;
            ln.x = map_x;
            ln.y = map_y;
            
            mapObservations.push_back(ln);
        }
        
        // Associate to each observation corresponding landmark
        dataAssociation(map_landmarks.landmark_list, mapObservations);
        
     // Now id of each observation is the corresponing index to the landmark in the map list
        double w = 1.0;
        
        // Compute probability
        
        for(int i = 0; i < mapObservations.size(); i++){
            LandmarkObs obs = mapObservations[i];
            Map::single_landmark_s landmark = map_landmarks.landmark_list[obs.id];
            
            double dx = obs.x - landmark.x_f;
            double dy = obs.y - landmark.y_f;
            
            double prob = exp( -(dx*dx/(2*sigmax*sigmax) +  dy*dy/(2*sigmay*sigmay)) )/(2*M_PI*sigmax*sigmay);
            
            w = w * prob;
        }
        
        // OK now update particle weight
        
        particles[ip].weight = w;
    }
    // Now we must normalize weigths
    
    double wsum = 0;
    
    for(int i = 0; i < particles.size(); i++)
        wsum += particles[i].weight;

    for(int i = 0; i < particles.size(); i++)
        particles[i].weight = particles[i].weight / wsum;
    double sigma_pos [3] = {0.3, 0.3, 0.01};
    if(debug)checkParticles("Normalized", 0.0, 0.0, 0.0, sigma_pos);
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // Create a weight arrauy
    
    int np = particles.size();

    vector<double> weights;
    for(int i = 0; i < np; i++){
        weights.push_back(particles[i].weight);
    }
    std::discrete_distribution<int> dd(weights.begin(), weights.end());

    
    // Compute maxw
    
   vector<Particle> newParticles;
    
    // Compute maximum
    for(int i = 0; i < np; i++){
        int ip = dd(gen);   // Get an index
        newParticles.push_back(particles[ip]);
    }

    particles = newParticles;
    
    double sigma_pos [3] = {0.3, 0.3, 0.01};
    
    if(debug)checkParticles("Resampled", 0.0, 0.0, 0.0, sigma_pos);
    
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

/** Utilities
 *
 */

// No need to be a method. But just here for easy use and reference
std::tuple<double, double> ParticleFilter::ComputeStats(std::vector<double> data){
    
    int numPoints = data.size();
    double sum = accumulate(begin(data), end(data), 0.0, plus<double>());
    double mean = sum/numPoints;
    
    double var = 0.0;
    for(int  i = 0; i < numPoints; i++ )
    {
        var += (data[i] - mean) * (data[i] - mean);
    }
    var /= numPoints;
    double sd = sqrt(var);
    
    return std::make_tuple(mean, sd);
    
}
 
/** transform transforms a point in particle coordinates tp one in map coordinates
 */

std::tuple<double, double> ParticleFilter::transform(Particle p, double x, double y){
    
    double newx = p.x + cos(p.theta)*x - sin(p.theta)*y;
    double newy = p.y + sin(p.theta)*x + cos(p.theta)*y;
    
    return std::make_tuple(newx, newy);
}



double ParticleFilter::Normalize(double value){
    return atan2(sin(value), cos(value));
    
}
 
 /** DEBUG
 *
 */

bool ParticleFilter::checkParticles(string msg, double x, double y, double theta, double std[]){
    
    vector<double> v_x;
    vector<double> v_y;
    vector<double> v_theta;
    
    double m_x;
    double sd_x;
    double m_y;
    double sd_y;
    double m_theta;
    double sd_theta;
    
    // Build vectors iterating over particles
    
    for (int i = 0; i < particles.size(); i++){
        Particle p = particles[i];
        v_x.push_back(p.x);
        v_y.push_back(p.y);
        v_theta.push_back(p.theta);
    }
    
    // Compute stats
    
    std::tie(m_x, sd_x) = ComputeStats(v_x);
    std::tie(m_y, sd_y) = ComputeStats(v_y);
    std::tie(m_theta, sd_theta) = ComputeStats(v_theta);
    
    // Compute total weight (shoukd be 1.0
    
    double w = 0;
    for(int i = 0; i < particles.size(); i++){
        w += particles[i].weight;
    }
    
    // Print differences with desired values
    
    cout <<endl <<  msg << endl;
    cout << "X: " << x << " Averaged: " << m_x << " Std: " << std[0] << " Generated: " << sd_x << endl;
    cout << "Y: " << y << " Averaged: " << m_y << " Std: " << std[1] << " Generated: " << sd_y << endl;
    cout << "Theta: " << theta << " Averaged: " << m_theta << " Std: " << std[2] << " Generated: " << sd_theta << endl;
    cout << "Weights: " << w << endl << endl;

    return true;
    
    
}

