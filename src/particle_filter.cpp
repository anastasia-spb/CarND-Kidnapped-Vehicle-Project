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

LandmarkObs transform_observation(double x_translation, double y_translation, double theta, const LandmarkObs& input)
{
	/**
	* Transform position from VEHICLE'S to MAP'S coordinate system
	*/

	LandmarkObs transformed_observation{};
	transformed_observation.x = x_translation + cos(theta) * input.x - sin(theta) * input.y;
	transformed_observation.y = y_translation + sin(theta) * input.x + cos(theta) * input.y;
	transformed_observation.id = input.id;

	return transformed_observation;
}

std::vector<LandmarkObs> select_landmarks(const std::vector<Map::single_landmark_s>& landmarks,
										  double x_ref,
										  double y_ref,
										  double sensor_range)
{
	/**
	* Filter out landmarks which are not within sensor range
	*/
	std::vector<LandmarkObs> filtered_landmarks;

	for (auto landmark : landmarks)
	{
		const auto landmark_dist = dist(x_ref, y_ref, landmark.x_f, landmark.y_f);
		if (landmark_dist < sensor_range) 
		{
			filtered_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
		}
	}

	return filtered_landmarks;
}

LandmarkObs get_nearest_landmark(const std::vector<LandmarkObs>& landmarks, const LandmarkObs& ref_observation)
{
	LandmarkObs nearest_landmark{};
	double distance_to_ref = 0.0;
	double min_distance_to_ref = std::numeric_limits<double>::max();

	for (const auto& landmark : landmarks)
	{
		distance_to_ref = dist(ref_observation.x, ref_observation.y, landmark.x, landmark.y);
		if (distance_to_ref < min_distance_to_ref) {
			min_distance_to_ref = distance_to_ref;
			nearest_landmark = landmark;
		}
	}

	return nearest_landmark;
}

double gaussian(double sigma_x, double sigma_y, double delta_x, double delta_y)
{
	//  Multi-variate Gaussian
	constexpr double norm = 1 / (2 * M_PI);

	const auto exponent = pow(delta_x, 2) / (2 * pow(sigma_x, 2)) + pow(delta_y, 2) / (2 * pow(sigma_y, 2));
	return (norm / (sigma_x * sigma_y)) * exp(-exponent);
}

double calculate_particle_weight(const Particle& particle, double sensor_range, double std_landmark[],
	const std::vector<LandmarkObs> &observations,
	const Map &map_landmarks)
{
	double weight = 1.0;

	auto nearest_landmarks = select_landmarks(map_landmarks.landmark_list, particle.x, particle.y, sensor_range);

	for (const auto& observation : observations)
	{
		auto observation_map = transform_observation(particle.x, particle.y, particle.theta, observation);
		auto nearest_landmark = get_nearest_landmark(nearest_landmarks, observation_map);

		// Update the weights of each particle using a multi-variate Gaussian distribution
		weight *= gaussian(std_landmark[0], std_landmark[1], (observation_map.x - nearest_landmark.x),
			(observation_map.y - nearest_landmark.y));
	}

	return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a mult-variate Gaussian 
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

	for (auto i = 0; i < particles.size(); ++i) 
	{
		particles[i].weight = calculate_particle_weight(particles[i], sensor_range, std_landmark, observations, map_landmarks);
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   * to their weight. 
   */
	vector<Particle> resampled_particles(num_particles);

	std::random_device rd;
	std::mt19937 gen(rd());
	auto index = rand() % num_particles;

	auto max_weight = *max_element(weights.begin(), weights.end());
	std::discrete_distribution<> dist(0.0, 2.0 * max_weight);

	double beta = 0.0;
	for (auto i = 0; i < num_particles; ++i) 
	{
		beta += dist(gen);
		while (weights[index] < beta) 
		{
			beta -= weights[index];
			index = (++index) % num_particles;
		}
		resampled_particles[i] = particles[index];
	}

	particles = resampled_particles;
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