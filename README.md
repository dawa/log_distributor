This is a high-throughput logs distributor that will act as an initial receiver of packets of log messages, where each packet could have multiple log messages.
The distributor receives log message packets from a number of agents that collect and transmit application/infrastructure logs.
The distributor fronts several collectors, each collector being assigned a relative weight (e.g. 0.4, 0.3, 0.1, 0.2) - assume that the weights add up to 1.0.
The distributor should route log message packets to collectors, so that eventually each collector analyzes a fraction of log messages roughly proportional to their relative weight.
