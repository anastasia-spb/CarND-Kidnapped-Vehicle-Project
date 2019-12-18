/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /* Set the number of particles */
  num_particles = 100;

  std::random_device rd{};
  std::mt19937 gen{rd()};

  std::normal_distribution<> x_dist{x, std[0]};
  std::normal_distribution<> y_dist{y, std[1]};
  std::normal_distribution<> theta_dist{theta, std[2]};

  /**
   * Initialize all particles to first position
   * (based on estimates of x, y, theta and their uncertainties
   * from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  Particle particle{};

  for(auto i = 0; i < num_particles; ++i)
  {
    particle.id = i;
    particle.x = x_dist(gen);
    particle.y = y_dist(gen);
    particle.theta = theta_dist(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void update_position(double velocity, double yaw_rate, double delta_t, Particle& particle)
{
  if (std::fabs(yaw_rate) > std::numeric_limits<double>::min())
  {
    particle.x += (velocity / yaw_rate)*(sin(particle.theta + yaw_rate*delta_t) - sin(particle.theta));
    particle.y += (velocity / yaw_rate)*(cos(particle.theta) - cos(particle.theta + yaw_rate*delta_t));
    particle.theta += yaw_rate*delta_t;
  }
  else
  {
    particle.x += velocity * delta_t * cos(particle.theta);
    particle.y += velocity * delta_t * sin(particle.theta);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate)
{
  std::default_random_engine gen;

  auto x_dist = [&std_pos](double mean){return std::normal_distribution<>{mean, std_pos[0]};};
  auto y_dist = [&std_pos](double mean){return std::normal_distribution<>{mean, std_pos[1]};};
  auto theta_dist = [&std_pos](double mean){return std::normal_distribution<>{mean, std_pos[2]};};

  std::for_each(particles.begin(), particles.end(), [&](Particle& particle)
  {
    // Add measurements
    update_position(velocity, yaw_rate, delta_t, particle);

    // Add random Gaussian noise
    particle.x += x_dist(particle.x)(gen);
    particle.y += y_dist(particle.y)(gen);
    particle.theta += theta_dist(particle.theta)(gen);
  });

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */

  const auto get_closest_predicted_measurement = [&predicted](const LandmarkObs& observation){
    std::vector<double> distances(predicted.size());
    std::transform(predicted.begin(), predicted.end(), distances.begin(), [&observation](const LandmarkObs& predicted_measurement){
          return dist(predicted_measurement.x, predicted_measurement.y, observation.x, observation.y);
      });
    const auto min_distance_idx = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
    return predicted.begin() + min_distance_idx;
  };

  for(auto& observation : observations)
  {
    observation.id = get_closest_predicted_measurement(observation)->id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}